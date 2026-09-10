"""Find causal prefixes that cannot encounter a selected embedding-row patch.

For a causal decoder, the residual predicting token k depends only on input
rows through k, inclusive. If none of those IDs is patched, changing only the
selected embedding rows cannot alter that residual. A tied-head patch can
still change its output logits. This is an input-dependency check, not a model
execution or evidence that any word probability actually changed.

The first target's prefix excludes the candidate word. Later targets include
its teacher-forced pieces, which is precisely when an input-side intervention
can begin to matter. Count identical causal prefixes once, not once for every
candidate or original/replacement domain alias.
"""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from .paired_word_cases import EOS, VOCAB_SIZE, _record, _write_json


def exposures(tokens, scored_rows, patched_ids):
    """Return patched IDs seen at or before each prediction row, never after."""
    if (len(tokens.shape) != 1 or np.any(tokens < 0)
            or np.any(tokens >= VOCAB_SIZE)):
        raise ValueError('expected one valid input-token sequence')
    if (not patched_ids or any(type(token) is not int or not 0 <= token < VOCAB_SIZE
                              for token in patched_ids)):
        raise ValueError('patch IDs must be nonempty valid integer token IDs')
    result = []
    for row in scored_rows:
        if type(row) is not int or not 0 <= row < len(tokens):
            raise ValueError('prediction row is outside the input sequence')
        prefix = tokens[:row + 1].astype('<i4')
        touched = sorted(set(prefix.tolist()) & set(patched_ids))
        result.append({'prediction_row': row, 'causal_prefix_length': row + 1,
                       'causal_prefix_sha256': hashlib.sha256(prefix.tobytes()).hexdigest(),
                       'patched_input_ids': touched,
                       'input_path_must_be_unchanged': not touched})
    return result


def analyze(cases_path, output):
    """Validate a frozen three-token suite and exclusively write its exposure map."""
    cases_path, output = Path(cases_path).resolve(), Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    case_record = _record(cases_path)
    cases = json.loads(cases_path.read_text())
    if (cases.get('format') != 'pluto-paired-word-cases-v1'
            or cases.get('vocab_size') != VOCAB_SIZE or cases.get('eos_token_id') != EOS):
        raise ValueError('expected frozen paired-word cases')
    count, length = cases['case_count'], cases['context_length']
    if (type(count) is not int or count < 1 or type(length) is not int
            or length < 3 or len(cases['cases']) != count):
        raise ValueError('invalid packed case dimensions')
    batch_record = _record(cases['packed_batch']['path'])
    if batch_record != cases['packed_batch'] or batch_record['bytes'] != 8 * count * length:
        raise ValueError('packed case size/hash mismatch')
    packed = np.fromfile(batch_record['path'], dtype='<i4').reshape(2, count, length)
    if (np.any(packed < 0) or np.any(packed >= VOCAB_SIZE)
            or not np.array_equal(packed[0, :, 1:], packed[1, :, :-1])):
        raise ValueError('invalid token IDs or shifted next-token alignment')
    patched_ids = sorted({token for case in cases['cases'] if case['kind'] == 'word'
                          for token in case['target_ids']})
    per_case, unique = [], defaultdict(dict)
    for index, case in enumerate(cases['cases']):
        prefix_length = case['prefix']['length']
        rows = [prefix_length - 1, prefix_length, prefix_length + 1]
        if (type(prefix_length) is not int or prefix_length < 1
                or rows[-1] >= length or case['case_index'] != index
                or case['scored_rows'] != rows or len(case['target_ids']) != 3
                or packed[1, index, rows].tolist() != case['target_ids']):
            raise ValueError('case prediction rows/targets do not match packed tokens')
        observed = exposures(packed[0, index], rows, patched_ids)
        if observed[0]['causal_prefix_sha256'] != case['prefix']['token_ids_sha256']:
            raise ValueError('case prefix hash mismatch')
        per_case.append({'case_index': index, 'kind': case['kind'], 'split': case['split'],
                         'context_id': case['context_id'], 'target': case['target'],
                         'predictions': observed})
        for position, item in enumerate(observed):
            group = (case['kind'], case['split'], position)
            key = (item['causal_prefix_length'], item['causal_prefix_sha256'])
            unique[group][key] = item
    groups = []
    for (kind, split, position), prefixes in sorted(unique.items()):
        values = list(prefixes.values())
        groups.append({'kind': kind, 'split': split, 'target_position': position,
                       'unique_causal_prefix_count': len(values),
                       'unchanged_input_path_count': sum(
                           x['input_path_must_be_unchanged'] for x in values)})
    for record in (case_record, batch_record):
        if _record(record['path']) != record:
            raise ValueError('input changed while computing exposures')
    result = {'format': 'pluto-paired-input-exposure-v1', 'cases': case_record,
              'packed_batch': batch_record, 'patched_ids': patched_ids,
              'implementation': _record(__file__), 'per_case': per_case, 'groups': groups,
              'interpretation': 'No patched causal-prefix ID implies unchanged residual '
                                'for an embedding-row-only patch in a causal decoder; '
                                'the tied output dictionary may still change logits.',
              'limitations': ['No model was run and no probability change was measured.',
                              'Exposure permits an input-side effect but does not prove one.',
                              'The guarantee requires all other weights/settings unchanged.']}
    _write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.cases, args.output)


if __name__ == '__main__':
    main()
