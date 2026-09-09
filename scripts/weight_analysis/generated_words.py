"""Find review candidates in actual native autoregressive token boundaries.

GPT-2 can spell ordinary words with several tokens. Being absent as a single
vocabulary entry is NOT evidence of a novel word, memorization, or a new concept.
This module reports a reproducible structural filter and leaves lexical
commonness/ordinary-compound judgments to separate, documented review.

No text is re-tokenized. All offsets are byte offsets in the concatenation of
the supplied native prompt tokens and generated tokens. Importing this module
does not read checkpoints, corpora, tokenizers, dictionaries, or model outputs.
"""

import argparse
import bisect
import hashlib
import json
from pathlib import Path
import re
import struct


LETTERS = re.compile(rb'[A-Za-z]+')
WHOLE_WORD = re.compile(rb' ?[A-Za-z]+\Z')


def _valid_ids(ids, vocabulary, name):
    if not isinstance(ids, (list, tuple)) or any(
            type(token) is not int or token not in vocabulary for token in ids):
        raise ValueError(name + ' must be a sequence of valid integer token IDs')
    return list(ids)


def _vocabulary(token_bytes, special_token_ids):
    if (not isinstance(token_bytes, dict) or not token_bytes or
            any(type(token) is not int or token < 0 or not isinstance(piece, bytes)
                or not piece for token, piece in token_bytes.items())):
        raise ValueError('token_bytes must map nonnegative integer IDs to nonempty bytes')
    specials = set(_valid_ids(list(special_token_ids), token_bytes, 'special_token_ids'))
    index = {}
    for token, piece in token_bytes.items():
        if token not in specials and WHOLE_WORD.fullmatch(piece):
            # ASCII lower() covers every mixed/title/upper/lowercase variant,
            # not just the case observed in this particular generation.
            spelling = piece[1:] if piece.startswith(b' ') else piece
            index.setdefault(spelling.lower(), []).append(token)
    for matches in index.values():
        matches.sort()
    return specials, index


def analyze_generated_words(prompt_token_ids, generated_token_ids, token_bytes,
                            *, special_token_ids=(), lexicon=None,
                            lexicon_source=None):
    """Enumerate ASCII letter-runs and flag complete generated multi-piece words.

    A structural candidate must be wholly generated, have at least two ACTUAL
    generated token occurrences contributing letters, have an observed right
    delimiter, and lack any whole-word vocabulary spelling (bare or with one
    leading ASCII space, ignoring ASCII case). Repeated occurrences of one ID
    count as separate pieces. Space-only pieces do not count.

    Conservatively reject runs adjoining a non-ASCII byte, digit, underscore,
    apostrophe, hyphen, or special-token span. This avoids treating the ASCII
    fragment of a UTF-8 word, identifier, contraction, or hyphenated compound as
    a new standalone word. Such runs are still reported, with rejection reasons.
    A trailing letter-run is incomplete even when the generation limit was hit.

    ``lexicon`` is an optional externally supplied collection of ASCII words.
    Its membership and all two-part splits with >=2 letters per part are only
    descriptive evidence. No lexicon hit/miss certifies frequency or novelty;
    missing splits do not rule out longer compounds, morphology, or proper
    names. The caller must document the source and perform independent review.
    """
    specials, whole_word_index = _vocabulary(token_bytes, special_token_ids)
    prompt_ids = _valid_ids(prompt_token_ids, token_bytes, 'prompt_token_ids')
    generated_ids = _valid_ids(generated_token_ids, token_bytes, 'generated_token_ids')
    if lexicon is None:
        lexical_words = None
        if lexicon_source is not None:
            raise ValueError('lexicon_source requires a supplied lexicon')
    else:
        if not isinstance(lexicon_source, str) or not lexicon_source:
            raise ValueError('a supplied lexicon needs a documented source')
        lexical_words = set()
        for word in lexicon:
            if not isinstance(word, str) or not re.fullmatch('[A-Za-z]+', word):
                raise ValueError('lexicon entries must be nonempty ASCII alphabetic words')
            lexical_words.add(word.lower())

    tokens, parts, cursor = [], [], 0
    for index, token in enumerate(prompt_ids + generated_ids):
        piece = token_bytes[token]
        generated = index >= len(prompt_ids)
        tokens.append({'token_index': index, 'token_id': token,
                       'origin': 'generated' if generated else 'prompt',
                       'generated_index': index-len(prompt_ids) if generated else None,
                       'byte_start': cursor, 'byte_end': cursor+len(piece),
                       'bytes_hex': piece.hex(),
                       'piece_escaped': repr(piece.decode('utf-8', errors='backslashreplace')),
                       'special': token in specials})
        parts.append(piece)
        cursor += len(piece)
    stream = b''.join(parts)
    prompt_bytes = sum(len(token_bytes[token]) for token in prompt_ids)
    ends = [token['byte_end'] for token in tokens]
    special_spans = [(token['byte_start'], token['byte_end']) for token in tokens
                     if token['special']]
    words = []
    for match in LETTERS.finditer(stream):
        start, end = match.span()
        first = bisect.bisect_right(ends, start)
        overlaps = []
        for token in tokens[first:]:
            if token['byte_start'] >= end:
                break
            left, right = max(start, token['byte_start']), min(end, token['byte_end'])
            overlaps.append({'token_index': token['token_index'], 'token_id': token['token_id'],
                             'origin': token['origin'], 'generated_index': token['generated_index'],
                             'byte_start': left, 'byte_end': right,
                             'token_byte_start': left-token['byte_start'],
                             'token_byte_end': right-token['byte_start'],
                             'word_byte_start': left-start, 'word_byte_end': right-start,
                             'bytes_hex': stream[left:right].hex(),
                             'token_crosses_word_start': token['byte_start'] < start,
                             'token_crosses_word_end': token['byte_end'] > end})
        assert b''.join(bytes.fromhex(overlap['bytes_hex']) for overlap in overlaps) == match.group()
        generated_overlaps = [overlap for overlap in overlaps if overlap['origin'] == 'generated']
        rejection = []
        complete = end < len(stream)
        if not complete:
            rejection.append('trailing_word_has_no_observed_right_delimiter')
        if start < prompt_bytes:
            rejection.append('word_is_not_wholly_generated')
        if len(generated_overlaps) < 2:
            rejection.append('fewer_than_two_generated_pieces_contribute_letters')
        touches_special = any(start < b and end > a or start == b or end == a
                              for a, b in special_spans)
        if touches_special:
            rejection.append('word_overlaps_or_touches_special_token')
        for side, position in [('left', start-1), ('right', end)]:
            if 0 <= position < len(stream):
                byte = stream[position]
                if byte >= 128:
                    rejection.append(side+'_boundary_is_non_ascii')
                elif 48 <= byte <= 57 or byte in b"_'-":
                    rejection.append(side+'_boundary_joins_identifier_or_compound')
        matches = whole_word_index.get(match.group().lower(), [])
        if matches:
            rejection.append('whole_word_exists_as_single_vocabulary_token')
        text = match.group().decode('ascii')
        lower = text.lower()
        splits = [] if lexical_words is None else [
            [lower[:i], lower[i:]] for i in range(2, len(lower)-1)
            if lower[:i] in lexical_words and lower[i:] in lexical_words]
        lexical = {'review_required': not rejection,
                   'status': 'unreviewed', 'lexicon_source': lexicon_source,
                   'whole_word_in_supplied_lexicon': None if lexical_words is None else lower in lexical_words,
                   'two_part_lexicon_splits': splits,
                   'warning': 'Single-token OOV and missing lexicon entries do not establish novelty or uncommonness.'}
        words.append({'word_index': len(words), 'word': text,
                      'byte_start': start, 'byte_end': end,
                      'generated_byte_start': start-prompt_bytes if start >= prompt_bytes else None,
                      'generated_byte_end': end-prompt_bytes if end > prompt_bytes else None,
                      'complete': complete, 'wholly_generated': start >= prompt_bytes,
                      'generated_piece_count': len(generated_overlaps), 'token_overlaps': overlaps,
                      'single_token_vocabulary_matches': [
                          {'id': token, 'bytes_hex': token_bytes[token].hex()} for token in matches],
                      'structural_candidate': not rejection,
                      'rejection_reasons': rejection, 'lexical_commonness': lexical})
    try:
        stream.decode('utf-8')
        valid_utf8 = True
    except UnicodeDecodeError:
        valid_utf8 = False
    return {'schema_version': 1, 'retokenized': False,
            'prompt_token_ids': prompt_ids, 'generated_token_ids': generated_ids,
            'prompt_byte_length': prompt_bytes, 'stream_byte_length': len(stream),
            'stream_bytes_hex': stream.hex(), 'valid_utf8': valid_utf8,
            'stream_escaped': repr(stream.decode('utf-8', errors='backslashreplace')),
            'tokens': tokens, 'words': words,
            'candidate_word_indices': [word['word_index'] for word in words if word['structural_candidate']],
            'lexical_review_complete': False,
            'filter': 'Complete wholly-generated ASCII word; >=2 actual generated pieces; no bare/one-leading-space case-insensitive whole-word vocabulary token.',
            'limits': ['Trailing partial words are not complete candidates.',
                       'Word fragments touching non-ASCII, identifiers, contractions, hyphens, or special tokens are conservatively excluded.',
                       'Vocabulary absence is not lexical novelty; commonness, morphology, compounds and names require separate review.',
                       'Two-part lexicon splits are descriptive, not an exhaustive compound detector.']}


def analyze_native_events(metadata, events, token_bytes, *, generated_bytes=None,
                          lexicon=None, lexicon_source=None):
    """Authenticate word boundaries against recorded native IDs/byte events.

    This checks byte/event consistency, not RNG arithmetic or model logits.
    Those require separate audits. It never re-encodes the generated text.
    """
    if metadata.get('complete') is not True:
        raise ValueError('native generation metadata is not complete')
    initial = metadata.get('initial_token_ids')
    generated = metadata.get('generated_token_ids')
    report = analyze_generated_words(initial, generated, token_bytes,
        special_token_ids=metadata.get('special_token_ids', ()),
        lexicon=lexicon, lexicon_source=lexicon_source)
    all_ids = _valid_ids(metadata.get('all_token_ids'), token_bytes, 'all_token_ids')
    if (all_ids != initial+generated
            or type(metadata.get('steps')) is not int or metadata['steps'] != len(generated)
            or not isinstance(events, list) or len(events) != len(generated)):
        raise ValueError('native token/event counts or concatenated IDs disagree')
    prompt = b''.join(token_bytes[token] for token in initial)
    if not isinstance(metadata.get('prompt'), str) or metadata['prompt'].encode('utf-8') != prompt:
        raise ValueError('native prompt bytes disagree with initial token IDs')
    cursor = 0
    for index, (token, event) in enumerate(zip(generated, events)):
        expected = {'index': index, 'absolute_token_index': len(initial)+index,
                    'token_id': token, 'generated_byte_start': cursor,
                    'generated_byte_end': cursor+len(token_bytes[token])}
        if (not isinstance(event, dict) or any(type(event.get(key)) is not int or event[key] != value
                                              for key, value in expected.items())
                or event.get('piece_hex') != token_bytes[token].hex()):
            raise ValueError('native token event has inconsistent identity or byte offsets')
        cursor += len(token_bytes[token])
    actual_generated = b''.join(token_bytes[token] for token in generated)
    if generated_bytes is not None and generated_bytes != actual_generated:
        raise ValueError('native generated.bin differs from event/token bytes')
    report['native_event_bytes_verified'] = True
    report['sampling_or_logits_audited'] = False
    return report


def _identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(1024*1024):
            digest.update(block)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def run(native_directory, tokenizer_directory, output_path):
    """CLI adapter; preserve exact input/source identities before and after."""
    from tokenizers import Tokenizer
    from .verify import gpt2_token_bytes

    native = Path(native_directory).resolve()
    output = Path(output_path)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    metadata_path = native/'metadata.json'
    tokenizer_path = Path(tokenizer_directory).resolve()/'tokenizer.json'
    records = {name: _identity(path) for name, path in {
        'metadata': metadata_path, 'tokenizer': tokenizer_path,
        'source': Path(__file__), 'byte_decoder_source': Path(__file__).with_name('verify.py')}.items()}
    metadata = json.loads(metadata_path.read_text())
    if Path(metadata['tokenizer_directory']).resolve() != Path(tokenizer_directory).resolve():
        raise ValueError('tokenizer directory differs from native run')

    def native_file(name):
        filename = Path(metadata['files'][name]['file'])
        if filename.is_absolute() or len(filename.parts) != 1:
            raise ValueError('native artifact must be a local filename')
        path = native/filename
        if path.is_symlink() or not path.is_file():
            raise ValueError('native artifact must be a regular nonsymlink file')
        records[name] = _identity(path)
        return path

    events = [json.loads(line) for line in native_file('events').read_text().splitlines()]
    raw_generated = native_file('generated_bytes').read_bytes()
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokens = gpt2_token_bytes(tokenizer)
    if type(metadata.get('vocab_size')) is not int or metadata['vocab_size'] != len(tokens):
        raise ValueError('native vocabulary size differs from supplied tokenizer')
    result = analyze_native_events(metadata, events, tokens, generated_bytes=raw_generated)
    ids = metadata['all_token_ids']
    token_record = metadata['files']['tokens']
    if token_record.get('dtype') != 'int32' or token_record.get('shape') != [len(ids)]:
        raise ValueError('native token artifact must be a flat int32 ID array')
    token_data = native_file('tokens').read_bytes()
    if len(token_data) != 4*len(ids) or list(struct.unpack('<'+'i'*len(ids), token_data)) != ids:
        raise ValueError('native token artifact disagrees with recorded token IDs')
    for record in records.values():
        if _identity(record['path']) != record:
            raise ValueError('native generation input changed during candidate discovery')
    result['input_records'] = records
    result['all_checked_inputs_unchanged'] = True
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-directory', type=Path, required=True)
    parser.add_argument('--tokenizer-directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.native_directory, args.tokenizer_directory, args.output)
    print(f"Enumerated {len(result['words'])} word runs; "
          f"{len(result['candidate_word_indices'])} structural candidates need lexical review.")


if __name__ == '__main__':
    main()
