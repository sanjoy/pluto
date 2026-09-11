"""Find every corpus occurrence of already-generated structural word candidates.

This is a read-only corpus lookup, not a generator, unfamiliarity classifier,
target selector, or claim of memorization. Native token/byte boundaries are
validated instead of re-tokenizing matching words. A matching spelling need
not have the same token IDs, capitalization, or surrounding whitespace.
"""

from __future__ import annotations

import argparse
import bisect
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import unicodedata

import numpy as np

from . import causal_validation as cv
from . import generated_words as generated
from .verify import gpt2_token_bytes, validate_native_tokens
from .vocabulary_verify import checked_tokens


BOUNDARY_POLICY = (
    'ASCII-case-insensitive spelling; adjacent Unicode alphanumerics, combining marks, '
    'underscore, apostrophes or hyphens exclude a whole-word match. Other Unicode '
    'punctuation and corpus edges delimit words. No substring or compound inference.')
JOINERS = frozenset("_'\u2019\u02bc-\u2010\u2011")


def _same(actual, expected, label):
    # JSON comparison distinguishes bools from equal-valued integers too.
    if json.dumps(actual, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
        raise ValueError(label + ' differs from exact generated token bytes')


def validate_generated_report(report, vocabulary):
    """Recheck structural candidates from their original token IDs, not encoding.

    Lexical-review annotations are deliberately excluded: this tool neither
    trusts them as unfamiliarity labels nor changes the structural candidate set.
    CLI provenance checks additionally bind these IDs to the native run.
    """
    if not isinstance(report, dict) or type(report.get('schema_version')) is not int or report['schema_version'] != 1:
        raise ValueError('expected generated_words schema_version=1')
    token_rows = report.get('tokens')
    if not isinstance(token_rows, list) or any(not isinstance(row, dict) for row in token_rows):
        raise ValueError('missing generated token records')
    specials = {row['token_id'] for row in token_rows if row.get('special') is True}
    rebuilt = generated.analyze_generated_words(
        report.get('prompt_token_ids'), report.get('generated_token_ids'), vocabulary,
        special_token_ids=specials)
    for key in ('retokenized', 'prompt_token_ids', 'generated_token_ids', 'prompt_byte_length',
                'stream_byte_length', 'stream_bytes_hex', 'valid_utf8', 'tokens', 'candidate_word_indices'):
        _same(report.get(key), rebuilt[key], key)
    if rebuilt['valid_utf8'] is not True:
        raise ValueError('generated stream is not complete valid UTF-8')
    if not isinstance(report.get('words'), list) or len(report['words']) != len(rebuilt['words']):
        raise ValueError('generated word count differs')
    for old, new in zip(report['words'], rebuilt['words']):
        if not isinstance(old, dict):
            raise ValueError('malformed generated word record')
        _same({key: old.get(key) for key in new if key != 'lexical_commonness'},
              {key: value for key, value in new.items() if key != 'lexical_commonness'}, 'generated word')
    return rebuilt


def _adjacent(corpus, position, left):
    """Decode exactly the neighboring codepoint at a known ASCII-word boundary."""
    if left:
        if position == 0:
            return None
        start = position - 1
        while start and corpus[start] & 0xc0 == 0x80:
            start -= 1
        return corpus[start:position].decode('utf-8')
    if position == len(corpus):
        return None
    end = position + 1
    while end < len(corpus) and corpus[end] & 0xc0 == 0x80:
        end += 1
    return corpus[position:end].decode('utf-8')


def _joined(character):
    return character is not None and (character.isalnum() or character in JOINERS
                                     or unicodedata.category(character).startswith('M'))


def whole_word_spans(corpus, word):
    """Exact byte ranges with independently checked UTF-8/whole-word boundaries."""
    if not isinstance(corpus, bytes):
        raise ValueError('corpus must be raw bytes')
    corpus.decode('utf-8')  # Reject invalid text; do not hide it with replacement characters.
    if not isinstance(word, str) or re.fullmatch(r'[A-Za-z]+', word) is None:
        raise ValueError('candidate must be a nonempty ASCII alphabetic word')
    result = []
    for match in re.finditer(re.escape(word.encode('ascii')), corpus, flags=re.IGNORECASE | re.ASCII):
        start, end = match.span()
        left, right = _adjacent(corpus, start, True), _adjacent(corpus, end, False)
        if not _joined(left) and not _joined(right):
            result.append((start, end, left, right))
    return result


def partition(start, end, split_byte):
    """Membership concerns exact byte spans, not assumed historical examples."""
    if end <= split_byte:
        return 'training_prefix'
    if start >= split_byte:
        return 'heldout_suffix'
    return 'crosses_split'


def analyze(report, corpus, tokens, offsets, vocabulary, *, split_byte=None):
    """Report all candidates and all matches, including zero-hit candidates.

    A custom split may cut an ASCII word or a native token; those cases are
    reported rather than silently rounded to a token boundary. It must not cut
    a UTF-8 codepoint. Without an override, use today's corpus split policy.
    """
    if not isinstance(corpus, bytes):
        raise ValueError('corpus must be raw bytes')
    corpus.decode('utf-8')
    rebuilt = validate_generated_report(report, vocabulary)
    if set(vocabulary) != set(range(len(vocabulary))):
        raise ValueError('native vocabulary IDs must be contiguous from zero')
    tokens = checked_tokens(tokens, len(vocabulary))
    offsets = np.asarray(offsets)
    if offsets.ndim != 1 or offsets.dtype.kind not in 'iu' or np.any(offsets < 0):
        raise ValueError('native byte offsets must be unsigned-range integer values')
    validate_native_tokens(tokens, offsets, corpus, vocabulary)
    default_split = split_byte is None
    if default_split:
        split_byte = cv.split_boundary(corpus)
    if type(split_byte) is not int or not 0 <= split_byte <= len(corpus):
        raise ValueError('split byte must lie in [0, corpus size]')
    if split_byte < len(corpus) and corpus[split_byte] & 0xc0 == 0x80:
        raise ValueError('split byte cuts a UTF-8 codepoint')
    newline_positions = [match.start() for match in re.finditer(b'\n', corpus)]
    cached_spans = {}
    candidates = []
    for word_index in rebuilt['candidate_word_indices']:
        word = rebuilt['words'][word_index]
        spelling = word['word']
        generated_ids = [item['token_id'] for item in word['token_overlaps']]
        generated_clipped = [bytes.fromhex(item['bytes_hex']) for item in word['token_overlaps']]
        generated_full = [vocabulary[token] for token in generated_ids]
        folded = spelling.lower()
        if folded not in cached_spans:
            cached_spans[folded] = whole_word_spans(corpus, spelling)
        occurrences = []
        for start, end, left, right in cached_spans[folded]:
            first = int(np.searchsorted(offsets, start, side='right')) - 1
            stop = int(np.searchsorted(offsets, end, side='left'))
            native_ids = tokens[first:stop].tolist()
            overlaps = []
            for index in range(first, stop):
                token_start, token_end = int(offsets[index]), int(offsets[index + 1])
                overlap_start, overlap_end = max(start, token_start), min(end, token_end)
                overlaps.append(dict(
                    token_index=index, token_id=int(tokens[index]),
                    token_byte_start=token_start, token_byte_end=token_end,
                    full_piece_hex=corpus[token_start:token_end].hex(),
                    overlap_byte_start=overlap_start, overlap_byte_end=overlap_end,
                    offset_within_token_start=overlap_start-token_start,
                    offset_within_token_end=overlap_end-token_start,
                    offset_within_word_start=overlap_start-start,
                    offset_within_word_end=overlap_end-start,
                    overlap_piece_hex=corpus[overlap_start:overlap_end].hex(),
                    token_crosses_word_start=token_start < start,
                    token_crosses_word_end=token_end > end))
            clipped = [bytes.fromhex(item['overlap_piece_hex']) for item in overlaps]
            if b''.join(clipped) != corpus[start:end]:
                raise AssertionError('native overlaps do not reconstruct the matched word')
            line_index = bisect.bisect_left(newline_positions, start)
            line_start = newline_positions[line_index - 1] + 1 if line_index else 0
            span_start, span_end = int(offsets[first]), int(offsets[stop])
            occurrences.append(dict(
                byte_start=start, byte_end=end, spelling=corpus[start:end].decode('ascii'),
                exact_case_match=corpus[start:end] == spelling.encode('ascii'),
                line=line_index+1, byte_column=start-line_start+1,
                character_column=len(corpus[line_start:start].decode('utf-8'))+1,
                preceding_character=left, following_character=right,
                word_partition=partition(start, end, split_byte),
                native_token_span_partition=partition(span_start, span_end, split_byte),
                native_token_start=first, native_token_end=stop,
                native_span_byte_start=span_start, native_span_byte_end=span_end,
                native_token_ids=native_ids, native_token_overlaps=overlaps,
                native_token_ids_equal_generated_word_overlap_ids=native_ids == generated_ids,
                full_token_piece_bytes_equal_generated=[vocabulary[token] for token in native_ids] == generated_full,
                clipped_word_piece_bytes_equal_generated=clipped == generated_clipped,
                clipped_word_piece_ascii_casefold_equal_generated=
                    [piece.lower() for piece in clipped] == [piece.lower() for piece in generated_clipped]))
        counts = {name: sum(item['word_partition'] == name for item in occurrences)
                  for name in ('training_prefix', 'heldout_suffix', 'crosses_split')}
        candidates.append(dict(
            word_index=word_index, word=spelling, generated_byte_start=word['generated_byte_start'],
            generated_byte_end=word['generated_byte_end'], generated_stream_byte_start=word['byte_start'],
            generated_stream_byte_end=word['byte_end'], generated_token_overlaps=word['token_overlaps'],
            generated_word_overlap_token_ids=generated_ids, corpus_occurrence_count=len(occurrences),
            corpus_occurrences_by_partition=counts,
            exact_native_token_id_sequence_occurrence_count=sum(
                item['native_token_ids_equal_generated_word_overlap_ids'] for item in occurrences),
            unfamiliarity='not_assessed', occurrences=occurrences))
    prefix_complete = int(np.searchsorted(offsets[1:], split_byte, side='right'))
    suffix_complete = len(tokens) - int(np.searchsorted(offsets[:-1], split_byte, side='left'))
    return dict(
        schema_version=1, complete=True, stage='corpus_lookup_of_all_generated_structural_candidates',
        retokenized=False, native_export_every_byte_validated=True, corpus_utf8_valid=True,
        generated_stream_utf8_valid=True, boundary_policy=BOUNDARY_POLICY,
        corpus_bytes=len(corpus), corpus_tokens=len(tokens),
        split=dict(byte_boundary=split_byte, source='current_default_policy' if default_split else 'explicit_override',
                   is_native_token_boundary=bool(np.any(offsets == split_byte)),
                   complete_native_tokens_in_prefix=prefix_complete,
                   complete_native_tokens_in_suffix=suffix_complete,
                   native_tokens_crossing_split=len(tokens)-prefix_complete-suffix_complete),
        candidate_count=len(candidates), candidates_with_corpus_occurrences=sum(
            item['corpus_occurrence_count'] > 0 for item in candidates),
        candidates_with_training_prefix_occurrences=sum(
            item['corpus_occurrences_by_partition']['training_prefix'] > 0 for item in candidates),
        candidates=candidates, unfamiliarity_assessed=False, target_selected=False,
        limits=[
            'All structural candidates retained, including zero matches and repeated generated words.',
            'Corpus spelling presence is not evidence of unusualness, memorization, or causal retrieval.',
            'Native overlapping token IDs may differ with case/whitespace; full and clipped pieces are separate.',
            'An identical ID list is not a claim that surrounding native bytes contain only the word.',
            'Prefix/heldout membership describes the supplied current byte split, not authenticated historical training.'])


def _check_record(record):
    if not isinstance(record, dict) or set(record) != {'path', 'bytes', 'sha256'}:
        raise ValueError('malformed input identity')
    if cv.file_record(record['path']) != record:
        raise ValueError('input/source changed: ' + record['path'])


def run(generated_words, corpus, native_tokens, native_offsets, tokenizer_dir, output, *, split_byte=None):
    """Authenticate completed generator provenance and preserve lookup hashes."""
    from tokenizers import Tokenizer

    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    paths = dict(generated_words=Path(generated_words).resolve(), corpus=Path(corpus).resolve(),
                 native_tokens=Path(native_tokens).resolve(), native_offsets=Path(native_offsets).resolve(),
                 tokenizer=(Path(tokenizer_dir)/'tokenizer.json').resolve())
    sources = {name: cv.file_record(Path(__file__).with_name(name)) for name in
               ('corpus_word_candidates.py', 'generated_words.py', 'verify.py',
                'vocabulary_verify.py', 'causal_validation.py')}
    inputs = {name: cv.file_record(path) for name, path in paths.items()}
    report = cv.read_json(paths['generated_words'])
    if report.get('native_event_bytes_verified') is not True or report.get('all_checked_inputs_unchanged') is not True:
        raise ValueError('requires a completed native generated_words output')
    origin = report.get('input_records', {})
    if set(origin) != {'metadata', 'tokenizer', 'source', 'byte_decoder_source', 'events', 'generated_bytes', 'tokens'}:
        raise ValueError('missing completed generator provenance')
    for record in origin.values():
        _check_record(record)
    if origin['tokenizer'] != inputs['tokenizer']:
        raise ValueError('tokenizer differs from completed candidate generation')
    if (Path(origin['source']['path']) != Path(generated.__file__).resolve()
            or Path(origin['byte_decoder_source']['path']) != Path(__file__).with_name('verify.py').resolve()):
        raise ValueError('generated candidate source/decoder provenance was substituted')
    tokenizer = Tokenizer.from_file(str(paths['tokenizer']))
    vocabulary = gpt2_token_bytes(tokenizer)
    metadata = cv.read_json(origin['metadata']['path'])
    if (type(metadata.get('vocab_size')) is not int or metadata['vocab_size'] != len(vocabulary)
            or not isinstance(metadata.get('tokenizer_directory'), str)
            or Path(metadata['tokenizer_directory']).resolve() != paths['tokenizer'].parent):
        raise ValueError('native generation tokenizer/vocabulary metadata differs')
    events = [json.loads(line) for line in Path(origin['events']['path']).read_text().splitlines()]
    native_report = generated.analyze_native_events(
        metadata, events, vocabulary, generated_bytes=Path(origin['generated_bytes']['path']).read_bytes())
    for key in ('prompt_token_ids', 'generated_token_ids', 'tokens', 'candidate_word_indices'):
        _same(report.get(key), native_report[key], 'native origin ' + key)
    native_ids_bytes = Path(origin['tokens']['path']).read_bytes()
    if (len(native_ids_bytes) % 4 or np.frombuffer(native_ids_bytes, dtype='<i4').tolist()
            != metadata['all_token_ids']):
        raise ValueError('original native token file differs from generated IDs')
    if inputs['native_tokens']['bytes'] % 4 or inputs['native_offsets']['bytes'] % 8:
        raise ValueError('trailing partial native token/offset element')
    result = analyze(report, paths['corpus'].read_bytes(),
                     np.fromfile(paths['native_tokens'], dtype='<u4'),
                     np.fromfile(paths['native_offsets'], dtype='<u8'), vocabulary, split_byte=split_byte)
    for record in [*sources.values(), *inputs.values(), *origin.values()]:
        _check_record(record)
    result.update(completed_utc=datetime.now(timezone.utc).isoformat(), sources=sources,
                  inputs=inputs, generated_origin_records=origin, all_checked_inputs_unchanged=True)
    serialized = json.dumps(result, indent=2, allow_nan=False) + '\n'
    with output.open('x') as stream:
        stream.write(serialized)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('generated-words', 'corpus', 'native-tokens', 'native-offsets', 'tokenizer-dir', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--split-byte', type=int,
                        help='Override current default byte split; may cut a native token but not a UTF-8 codepoint.')
    args = parser.parse_args(argv)
    result = run(args.generated_words, args.corpus, args.native_tokens, args.native_offsets,
                 args.tokenizer_dir, args.output, split_byte=args.split_byte)
    print(f"Checked all {result['candidate_count']} structural candidates; "
          f"{result['candidates_with_training_prefix_occurrences']} have current-prefix occurrences. "
          'Unfamiliarity remains unassessed.')


if __name__ == '__main__':
    main()
