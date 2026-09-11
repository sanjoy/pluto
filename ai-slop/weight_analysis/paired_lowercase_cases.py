"""Preserve old word tests while adding the real lowercase replacement cases.

This reads the independently prepared lowercase amendment; it never modifies an
experiment, corpus, checkpoint, or old case suite. Existing uppercase selections
are matched by token_start, because inserting lowercase alignment records shifts
occurrence indices. Their original order is preserved. Replacement prefixes are
rebuilt from amended native IDs, not copied from the outdated packed suite.

Every real lowercase training occurrence is appended with two prefix domains
and two case-matching candidates. Exact spelling and source domain stay separate
from the word family. Inherited controls must remain outside EVERY amended
replacement span; an overlapping control is an error, never silently reused or
resampled. No learned weights or model predictions affect case selection.
"""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from . import checkpoint, paired_lowercase_inputs, paired_training, paired_word_cases


FORMAT = 'pluto-paired-lowercase-word-cases-v1'
PAIRS = {'title': {'original': 'Exeunt', 'replacement': 'Nuveth'},
         'lowercase': {'original': 'exeunt', 'replacement': 'nuveth'}}


def _record(path):
    if Path(path).is_symlink():
        raise ValueError('case provenance must not name a symlink')
    return paired_word_cases._record(path)


def _require_record(expected, used):
    actual = _record(expected['path'])
    if actual != expected:
        raise ValueError('input/provenance hash mismatch: ' + expected['path'])
    old = used.setdefault(actual['path'], actual)
    if old != actual:
        raise ValueError('input changed during case preparation')
    return Path(actual['path'])


def _prefix(data, begin, start):
    tokens = data['tokens'][begin:start]
    return dict(token_start=begin, token_end=start, length=len(tokens),
                byte_start=int(data['offsets'][begin]), byte_end=int(data['offsets'][start]),
                token_ids_sha256=hashlib.sha256(tokens.astype('<i4').tobytes()).hexdigest())


def _same_native(a, b):
    return (a['text'] == b['text'] and np.array_equal(a['tokens'], b['tokens'])
            and np.array_equal(a['offsets'], b['offsets']))


def _variant(occurrence):
    pair = dict(original=occurrence['source_spelling'], replacement=occurrence['target_spelling'])
    matches = [name for name, expected in PAIRS.items() if pair == expected]
    if len(matches) != 1:
        raise ValueError('occurrence has a wrong case-matching candidate pair')
    return matches[0], pair


def _load_and_validate(amendment, old_manifest, used):
    """Recompute both alignments and old/new equality rather than trusting flags."""
    names = {f'{domain}.{split}' for domain in paired_word_cases.DOMAINS
             for split in ('full', *paired_word_cases.SPLITS)}
    if set(amendment['inputs']) != names or set(old_manifest['inputs']) != names:
        raise ValueError('need all six native full/training/test exports')
    old = {name: paired_word_cases._load_input(item, used)
           for name, item in old_manifest['inputs'].items()}
    new = {name: paired_word_cases._load_input(item, used)
           for name, item in amendment['inputs'].items()}
    for split in ('full', *paired_word_cases.SPLITS):
        a, b = (old[f'{domain}.{split}'] for domain in paired_word_cases.DOMAINS)
        if paired_training.verify_alignment(a['text'], b['text'], a['tokens'], b['tokens'],
                a['offsets'], b['offsets']) != old_manifest['alignment'][split]:
            raise ValueError('old uppercase-only alignment disagrees with native exports')
        original, replacement = (new[f'{domain}.{split}'] for domain in paired_word_cases.DOMAINS)
        if not _same_native(original, a):
            raise ValueError('amendment changed original native corpus data')
        verified = paired_lowercase_inputs.verify_alignment(original['text'], replacement['text'],
            original['tokens'], replacement['tokens'], original['offsets'], replacement['offsets'])
        if verified != amendment['alignment'][split]:
            raise ValueError('amended case spelling/alignment disagrees with native exports')
        lower_tokens = np.zeros(len(replacement['tokens']), dtype=bool)
        lower_boundaries = np.zeros(len(replacement['offsets']), dtype=bool)
        for occurrence in verified['occurrences']:
            variant, _ = _variant(occurrence)
            if variant == 'lowercase':
                start = occurrence['token_start']
                lower_tokens[start:start+3] = True
                lower_boundaries[start+1:start+3] = True
                # Actual lower-case occurrences have a leading-space native
                # triple. Bare exeunt is two tokens; never invent that variant.
                for domain, word in PAIRS['lowercase'].items():
                    source = paired_word_cases._target_source(new[f'{domain}.{split}'], start)
                    if bytes.fromhex(source['bytes_hex']) != b' ' + word.encode():
                        raise ValueError('lowercase occurrence is not a native leading-space triple')
        if (len(b['tokens']) != len(replacement['tokens'])
                or not np.array_equal(b['tokens'][~lower_tokens], replacement['tokens'][~lower_tokens])
                or not np.array_equal(b['offsets'][~lower_boundaries], replacement['offsets'][~lower_boundaries])):
            raise ValueError('amendment changed an old uppercase or unrelated token/boundary')
    boundary = amendment['split_byte']
    if type(boundary) is not int or boundary <= 0 or boundary != old_manifest['split_byte']:
        raise ValueError('amendment changed corpus split')
    for bundle in (old, new):
        for domain in paired_word_cases.DOMAINS:
            full, train, test = (bundle[f'{domain}.{split}'] for split in ('full', 'training', 'test'))
            if (len(train['text']) != boundary or not test['text']
                    or train['text']+test['text'] != full['text']
                    or not np.array_equal(np.r_[train['tokens'], test['tokens']], full['tokens'])
                    or not np.array_equal(np.r_[train['offsets'], test['offsets'][1:]+boundary], full['offsets'])):
                raise ValueError('full/split native exports disagree')
    return old, new


def prepare(amendments_path, old_cases_path, output, *, context_length=1024, prefix_tokens=128):
    """Exclusively create an amended case suite after validating all inputs.

    Small explicit geometry is allowed for CPU fixtures. Production defaults
    require the original 1024-row packing and 128-token maximum prefix; sequence
    positions still start at zero, exactly as in the preserved case suite.
    """
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    paired_word_cases._integer(context_length, 'context_length', 3)
    paired_word_cases._integer(prefix_tokens, 'prefix_tokens')
    if prefix_tokens > context_length-2:
        raise ValueError('prefix must leave room for three target rows')
    sources = [_record(module.__file__) for module in
               (checkpoint, paired_training, paired_word_cases, paired_lowercase_inputs)] + [_record(__file__)]
    amendment_record, old_cases_record = _record(amendments_path), _record(old_cases_path)
    used = {item['path']: item for item in (amendment_record, old_cases_record)}
    amendment = json.loads(Path(amendments_path).read_text())
    old_cases = json.loads(Path(old_cases_path).read_text())
    expected_rules = [dict(source=source.decode(), target=target.decode())
                      for source, target in paired_lowercase_inputs.REPLACEMENTS]
    if (amendment.get('format') != 'pluto-paired-lowercase-amendment-v1'
            or amendment.get('complete') is not True
            or amendment.get('replacement_rules') != expected_rules
            or amendment.get('old_word_cases') != old_cases_record
            or old_cases.get('format') != 'pluto-paired-word-cases-v1'
            or old_cases.get('manifest') != amendment.get('old_manifest')
            or old_cases.get('candidate_words') != PAIRS['title']
            or old_cases.get('context_length') != context_length
            or old_cases.get('selection', {}).get('prefix_tokens') != prefix_tokens
            or old_cases.get('vocab_size') != paired_word_cases.VOCAB_SIZE
            or old_cases.get('eos_token_id') != paired_word_cases.EOS):
        raise ValueError('amendment/old-case identity, candidate pair, or geometry mismatch')
    for item in amendment['provenance'] + amendment['sources']:
        _require_record(item, used)
    old_manifest_path = _require_record(amendment['old_manifest'], used)
    old_manifest = json.loads(old_manifest_path.read_text())
    if (old_manifest.get('format') != 'pluto-paired-corpus-training-v1'
            or old_manifest.get('replacement') != {'from': 'Exeunt', 'to': 'Nuveth', 'case_sensitive': True}):
        raise ValueError('old manifest is not the uppercase-only experiment')
    old_inputs, inputs = _load_and_validate(amendment, old_manifest, used)
    expected_old_sources = sorted({_record(old_manifest['inputs'][f'{domain}.{split}'][key])['path']
        for domain in paired_word_cases.DOMAINS for split in paired_word_cases.SPLITS
        for key in ('text', 'token_ids', 'offsets')})
    if (sorted(item['path'] for item in old_cases['source_inputs']) != expected_old_sources
            or len(old_cases['source_inputs']) != len(expected_old_sources)):
        raise ValueError('old cases reference a different native source set')
    for item in old_cases['source_inputs']:
        _require_record(item, used)
    count = paired_word_cases._integer(old_cases['case_count'], 'old case_count')
    batch_path = _require_record(old_cases['packed_batch'], used)
    if len(old_cases['cases']) != count or batch_path.stat().st_size != 8*count*context_length:
        raise ValueError('old packed case count/size mismatch')
    packed = np.fromfile(batch_path, dtype='<i4').reshape(2, count, context_length)
    if (np.any(packed < 0) or np.any(packed >= paired_word_cases.VOCAB_SIZE)
            or not np.array_equal(packed[0, :, 1:], packed[1, :, :-1])):
        raise ValueError('old packed teacher forcing is invalid')
    by_start, masks = {}, {}
    for split in paired_word_cases.SPLITS:
        occurrences = amendment['alignment'][split]['occurrences']
        by_start[split] = {o['token_start']: (i, o) for i, o in enumerate(occurrences)}
        if len(by_start[split]) != len(occurrences):
            raise ValueError('duplicate amended occurrence token_start')
        masks[split] = np.zeros(len(inputs[f'original.{split}']['tokens']), dtype=bool)
        for o in occurrences:
            masks[split][o['token_start']:o['token_start']+3] = True
    cases, xs, ys, selections = [], [], [], {}
    old_crossings, old_controls = defaultdict(set), set()

    def append(split, start, domain, target_domain, *, occurrence=None, occurrence_index=None,
               source_case=None):
        kind = 'word' if occurrence is not None else 'control'
        variant, pair = _variant(occurrence) if occurrence is not None else (None, None)
        prefix_domain = 'original' if domain == 'shared' else domain
        begin = max(0, start-prefix_tokens)
        data, target_data = inputs[f'{prefix_domain}.{split}'], inputs[f'{target_domain}.{split}']
        prefix = data['tokens'][begin:start]
        targets = target_data['tokens'][start:start+3].tolist()
        x, y, rows = paired_word_cases._case_sequence(prefix, targets, context_length)
        case = dict(case_index=len(cases), kind=kind, split=split, spelling_variant=variant,
            candidate_pair=pair, occurrence_index=occurrence_index,
            context_id=f'{split}:token_start:{start}:' + (variant or 'control'),
            prefix_domain=domain, prefix=_prefix(data, begin, start),
            target=pair[target_domain] if pair else 'control_next_3',
            target_source_domain=target_domain, target_ids=targets,
            target_source=paired_word_cases._target_source(target_data, start), scored_rows=rows,
            source_case_index=source_case['case_index'] if source_case else None,
            source_occurrence_index=source_case['occurrence_index'] if source_case else None,
            source_context_id=source_case['context_id'] if source_case else None)
        cases.append(case)
        xs.append(x)
        ys.append(y)

    # Preserve old case order, not just the selected set. This makes every old
    # score row visibly correspond to one new row, while changed prefixes are
    # honestly regenerated and re-hashed from the amended replacement corpus.
    for index, case in enumerate(old_cases['cases']):
        split, domain, target_domain = case['split'], case['prefix_domain'], case['target_source_domain']
        start, begin = case['prefix']['token_end'], case['prefix']['token_start']
        if (case['case_index'] != index or split not in paired_word_cases.SPLITS
                or target_domain not in paired_word_cases.DOMAINS
                or domain not in (*paired_word_cases.DOMAINS, 'shared')
                or type(start) is not int or begin != max(0, start-prefix_tokens)):
            raise ValueError('invalid inherited case identity or prefix coordinate')
        data = old_inputs[f'{"original" if domain == "shared" else domain}.{split}']
        target_data = old_inputs[f'{target_domain}.{split}']
        target = target_data['tokens'][start:start+3].tolist()
        x, y, rows = paired_word_cases._case_sequence(data['tokens'][begin:start], target, context_length)
        if (case['prefix'] != _prefix(data, begin, start)
                or case['target_source'] != paired_word_cases._target_source(target_data, start)
                or case['target_ids'] != target or case['scored_rows'] != rows
                or not np.array_equal(packed[0, index], x) or not np.array_equal(packed[1, index], y)):
            raise ValueError('inherited case differs from its native packed source')
        if case['kind'] == 'word':
            old_index = case['occurrence_index']
            old_occurrences = old_manifest['alignment'][split]['occurrences']
            if (type(old_index) is not int or not 0 <= old_index < len(old_occurrences)
                    or old_occurrences[old_index]['token_start'] != start
                    or domain == 'shared' or case['target'] != PAIRS['title'][target_domain]):
                raise ValueError('inherited uppercase selection or candidate label differs')
            new_index, occurrence = by_start[split][start]
            if _variant(occurrence)[0] != 'title':
                raise ValueError('old uppercase selection maps to lowercase after amendment')
            crossing = (domain, target_domain)
            if crossing in old_crossings[(split, old_index)]:
                raise ValueError('duplicate inherited word crossing')
            old_crossings[(split, old_index)].add(crossing)
            append(split, start, domain, target_domain, occurrence=occurrence,
                   occurrence_index=new_index, source_case=case)
        elif case['kind'] == 'control':
            if (domain != 'shared' or target_domain != 'original'
                    or case['target'] != 'control_next_3' or (split, start) in old_controls):
                raise ValueError('invalid or duplicate inherited control')
            if masks[split][begin:start+3].any():
                raise ValueError(f'inherited control overlaps an amended replacement span: '
                                 f'{split} prefix_start={begin} target_start={start}')
            if not np.array_equal(inputs[f'original.{split}']['tokens'][begin:start+3],
                                  inputs[f'replacement.{split}']['tokens'][begin:start+3]):
                raise ValueError('inherited control is no longer identical across corpora')
            old_controls.add((split, start))
            append(split, start, domain, target_domain, source_case=case)
            if not np.array_equal(xs[-1], packed[0, index]) or not np.array_equal(ys[-1], packed[1, index]):
                raise ValueError('inherited control packed bytes changed')
        else:
            raise ValueError('unsupported inherited case kind')
    expected_crossings = {(split, i) for split in paired_word_cases.SPLITS
        for i in old_cases['selection']['selected'][split]['occurrence_indices']}
    expected_controls = {(split, s) for split in paired_word_cases.SPLITS
        for s in old_cases['selection']['selected'][split]['control_token_starts']}
    full_cross = {(a, b) for a in paired_word_cases.DOMAINS for b in paired_word_cases.DOMAINS}
    if (set(old_crossings) != expected_crossings or any(v != full_cross for v in old_crossings.values())
            or old_controls != expected_controls):
        raise ValueError('inherited cases omit or add declared selections/crossings')
    lower_counts = {}
    for split in paired_word_cases.SPLITS:
        lower = [(i, o) for i, o in enumerate(amendment['alignment'][split]['occurrences'])
                 if _variant(o)[0] == 'lowercase']
        lower_counts[split] = len(lower)
        if split == 'test' and lower:
            raise ValueError('unexpected lowercase test occurrences in this amendment')
        selections[split] = dict(
            preserved_title_token_starts=sorted(old_manifest['alignment'][split]['occurrences'][i]['token_start']
                for s, i in old_crossings if s == split),
            preserved_control_token_starts=sorted(start for s, start in old_controls if s == split),
            added_lowercase_token_starts=[o['token_start'] for _, o in lower],
            lowercase_case_coverage='all native occurrences' if lower else 'no occurrences; no cases invented')
        for i, occurrence in lower:
            for domain in paired_word_cases.DOMAINS:
                for target_domain in paired_word_cases.DOMAINS:
                    append(split, occurrence['token_start'], domain, target_domain,
                           occurrence=occurrence, occurrence_index=i)
    if not lower_counts['training'] or len(cases) != count+4*sum(lower_counts.values()):
        raise ValueError('missing lowercase cases or unexpected total case count')
    piece_ids = sorted({token for split in paired_word_cases.SPLITS
        for occurrence in amendment['alignment'][split]['occurrences']
        for domain in paired_word_cases.DOMAINS for token in occurrence[f'{domain}_ids']})
    for item in [*used.values(), *sources]:
        if _record(item['path']) != item:
            raise ValueError('input or source changed during case preparation')
    result = dict(format=FORMAT, complete=True, case_count=len(cases),
        context_length=context_length, vocab_size=paired_word_cases.VOCAB_SIZE,
        eos_token_id=paired_word_cases.EOS,
        packing='little-endian int32 [all inputs case_count x context_length] [all targets case_count x context_length]',
        amendment=amendment_record, old_manifest=amendment['old_manifest'],
        source_word_cases=old_cases_record, source_packed_batch=old_cases['packed_batch'],
        candidate_pairs=PAIRS, word_piece_ids=piece_ids, cases=cases,
        selection=dict(model_outputs_used=False, algorithm='Preserve old selected starts/order; append every native lowercase training occurrence.',
            prefix_tokens=prefix_tokens, inherited_case_count=count,
            added_case_count=len(cases)-count, lowercase_occurrences=lower_counts, selected=selections),
        provenance=list(used.values()), sources=sources,
        checks=dict(original_native_exports_unchanged=True, old_case_order_preserved=True,
                    old_controls_bitwise_unchanged=True, all_amended_spans_excluded_from_controls=True,
                    actual_case_spellings_preserved=True, native_alignment_recomputed=True),
        limitations=[f'Capitalized contexts retain the old variant-stratified sample; all {lower_counts["training"]} real lowercase training contexts are included, with no lowercase test context available.',
                     'These are matched teacher-forced triples, not arbitrary lowercase tokenizations; bare exeunt has two tokens and is not invented as a case.',
                     'Prefix positions restart at zero and are not claimed to match historical training-window positions.',
                     'Replacement prefixes may change because lowercase replacements are now included; original prefixes and all controls remain unchanged.',
                     'No following delimiter is scored, no GPU work was performed, and no probabilities were measured by this preparation.'])
    output.mkdir()
    packed_path = output/'packed_cases.bin'
    with packed_path.open('xb') as stream:
        np.asarray(xs, dtype='<i4').tofile(stream)
        np.asarray(ys, dtype='<i4').tofile(stream)
    result['packed_batch'] = _record(packed_path)
    paired_word_cases._write_json(output/'cases.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--amendments', type=Path, required=True)
    parser.add_argument('--old-cases', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    result = prepare(args.amendments, args.old_cases, args.output)
    print(json.dumps(dict(cases=result['case_count'], word_piece_ids=result['word_piece_ids'],
                         selection=result['selection'])))


if __name__ == '__main__':
    main()
