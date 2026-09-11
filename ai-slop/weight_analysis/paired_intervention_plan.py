"""Prepare an immutable matched-step causal plan, without running any models.

A completed trajectory analysis is a prerequisite, not merely a file whose name
looks like a checkpoint. This planner revalidates its provenance and the actual
selected weights, selects the earliest/latest common positive optimizer steps,
and describes exact donor-byte interventions in both directions. It never
materializes checkpoint copies or executes a GPU process.

Case exports retain the original packing and case metadata while selecting
uniform three- or four-row targets. Their selected_rows.i32.bin is directly
consumable by embedding_factorial_probe; padding, tokens, and loss positions are
not regenerated or retokenized.
"""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import stat

import numpy as np

from . import checkpoint, paired_supplemental_cases, paired_word_cases
from . import paired_case_contract


FORMAT = 'pluto-paired-intervention-plan-v1'
ARMS = ('original', 'replacement')


def _record(path):
    return paired_word_cases._record(path)


def _verify_records(records):
    for record in records:
        if _record(record['path']) != record:
            raise ValueError(f'frozen provenance hash changed: {record["path"]}')


def _new_output(output, input_directories):
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    resolved = output.resolve()
    if any(resolved == Path(directory).resolve() or Path(directory).resolve() in resolved.parents
           for directory in input_directories):
        raise ValueError('output must be outside input directories')
    return output


def _case_input(path):
    record = _record(path)
    plan = json.loads(Path(path).read_text())
    if plan.get('format') not in paired_case_contract.LEGACY+paired_case_contract.AMENDED:
        raise ValueError('unsupported source case format')
    packed, packed_record = paired_supplemental_cases._packed(plan)
    for index, case in enumerate(plan['cases']):
        expected = (4,) if case['kind'] == 'word_next_native' else (3,)
        if case['kind'] not in ('word', 'control', 'word_next_native', 'shared_piece'):
            raise ValueError('unsupported source case kind')
        paired_supplemental_cases._validate_case(case, index, packed, expected)
    return plan, packed, [record, packed_record]


def export_cases(cases_path, output, *, indices=None, rows_per_case=None):
    """Copy selected cases in source order, preserving their exact input bytes.

    Indices must be unique, increasing, and nonempty. Every selected case must
    have the same three/four target count. Original source indices and hashes are
    recorded; source files are rehashed before publishing the completion JSON.
    """
    plan, packed, records = _case_input(cases_path)
    output = _new_output(output, (Path(cases_path).resolve().parent,
                                  Path(records[1]['path']).parent))
    selected = list(range(plan['case_count'])) if indices is None else list(indices)
    if (not selected or any(type(index) is not int or not 0 <= index < plan['case_count']
                            for index in selected) or selected != sorted(set(selected))):
        raise ValueError('case indices must be nonempty, unique, increasing, and in range')
    row_counts = {len(plan['cases'][index]['scored_rows']) for index in selected}
    if len(row_counts) != 1:
        raise ValueError('export must have one fixed target-row count, not mixed three/four rows')
    actual_rows = row_counts.pop()
    if actual_rows not in (3, 4) or (rows_per_case is not None and rows_per_case != actual_rows):
        raise ValueError('wrong requested rows_per_case')
    rows = np.asarray([plan['cases'][index]['scored_rows'] for index in selected], dtype='<i4')
    cases = [{**plan['cases'][index], 'case_index': new_index,
              'export_source_case_index': index} for new_index, index in enumerate(selected)]
    output.mkdir()
    with (output / 'packed_cases.bin').open('xb') as stream:
        # Advanced indexing preserves the [all inputs][all targets] layout.
        packed[:, selected, :].astype('<i4', copy=False).tofile(stream)
    with (output / 'selected_rows.i32.bin').open('xb') as stream:
        rows.tofile(stream)
    packed_record = _record(output / 'packed_cases.bin')
    rows_record = _record(output / 'selected_rows.i32.bin')
    exported = {**plan, 'case_count': len(selected), 'cases': cases, 'packed_batch': packed_record,
                'export_provenance': {'source_cases': records[0], 'source_packed_batch': records[1],
                                      'source_indices': selected, 'original_selection_metadata_preserved': True,
                                      'rows_per_case': actual_rows, 'selected_rows': rows_record},
                'exporter': _record(__file__)}
    _verify_records(records)
    paired_word_cases._write_json(output / 'cases.json', exported)
    return {'cases_json': _record(output / 'cases.json'), 'packed_batch': packed_record,
            'selected_rows': rows_record, 'rows_per_case': actual_rows,
            'case_count': len(selected), 'source_indices': selected,
            'source_cases': records[0], 'source_packed_batch': records[1]}


def _selected_pairs(summary):
    plan = summary['plan']
    if plan.get('all_matched_steps') is not True:
        raise ValueError('complete all-matched-step trajectory analysis is required')
    matched, endpoints = {}, None
    for pair in plan['pairs']:
        a, b = pair['original'], pair['replacement']
        if any(type(item['step']) is not int or item['step'] < 0 for item in (a, b)):
            raise ValueError('invalid optimizer step')
        if pair['name'] == 'final':
            if endpoints is not None:
                raise ValueError('duplicate final endpoint pair')
            endpoints = pair
        if a['step'] == b['step']:
            if a['step'] in matched and matched[a['step']] != pair:
                raise ValueError('duplicate matched-step pair')
            matched[a['step']] = pair
    if (endpoints is None or sorted(matched) != plan.get('matched_steps')
            or plan.get('matched_step') != max((step for step in matched if step > 0), default=None)):
        raise ValueError('matched-step inventory claims disagree with completed pairs')
    positive = sorted(step for step in matched if step > 0)
    if not positive:
        raise ValueError('no common positive optimizer step; unequal endpoints cannot substitute')
    return [matched[step] for step in sorted({positive[0], positive[-1]})], endpoints


def _verify_checkpoint(item, verified, root, arm, specs):
    path = Path(item['path'])
    if (path != root / arm / 'checkpoints' / f'step_{item["step"]}'
            or path.is_symlink() or not path.is_dir()):
        raise ValueError('selected checkpoint path does not match its arm and step')
    expected = {spec.filename: spec for spec in specs}
    if set(item['sha256']) != set(expected) or verified.get(str(path)) != item['sha256']:
        raise ValueError('checkpoint hashes disagree with completed analysis inventory')
    actual = {entry.name for entry in path.iterdir()}
    if actual != set(expected):
        raise ValueError('selected source checkpoint must contain only canonical weight files')
    for name, spec in expected.items():
        file = path / name
        info = file.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_size != spec.nbytes:
            raise ValueError(f'checkpoint weight layout mismatch: {file}')
        if checkpoint.sha256_file(file) != item['sha256'][name]:
            raise ValueError(f'checkpoint weight hash changed: {file}')


def _interventions(pair, specs, piece_ids):
    by_name = {spec.name: spec for spec in specs}
    block_count = sum(spec.name.endswith('.attn.qkv.weight') for spec in specs)
    entries = []
    for recipient, donor in (ARMS, ARMS[::-1]):
        step = pair[recipient]['step']

        def add(label, kind, names=(), rows=(), note=''):
            entries.append({'name': f'step_{step}_{donor}_to_{recipient}_{label}',
                            'step': step, 'recipient_arm': recipient, 'donor_arm': donor,
                            'recipient_checkpoint': pair[recipient], 'donor_checkpoint': pair[donor],
                            'tensors': list(names), 'embedding_rows': list(rows), 'kind': kind,
                            'selection_specs': [by_name[name].to_dict() for name in names],
                            'mechanism_note': note})

        add('copy_control', 'copy_control', note='Independent unmodified recipient copy; no donor bytes.')
        add('word_rows', 'embedding_rows', rows=piece_ids,
            note=f'{len(piece_ids)} selected embedding rows; changes input lookup and tied output readout jointly. '
                 'Use the factorial probe to separate lookup/readout effects.')
        for block in range(block_count):
            for branch, norm, projection in (('attention', 'ln1', 'attn'), ('mlp', 'ln2', 'mlp')):
                prefix = f'blocks.{block}'
                whole = [spec.name for spec in specs if spec.name.startswith(prefix + '.' + norm + '.')
                         or spec.name.startswith(prefix + '.' + projection + '.')]
                writes = [spec.name for spec in specs
                          if spec.name.startswith(prefix + '.' + projection + '.output.')]
                if len(whole) != 6 or len(writes) != 2:
                    raise ValueError('unexpected whole-branch/output-write tensor layout')
                add(f'block_{block}_{branch}_whole', f'{branch}_whole_branch', whole,
                    note='Transfers pre-LayerNorm, feature/query-key-value construction, and output '
                         'projection weights and biases; not a unique storage-location claim.')
                add(f'block_{block}_{branch}_output', f'{branch}_output_write', writes,
                    note='Transfers output projection weight AND bias only; recipient pre-norm and '
                         'feature construction remain unchanged. Not a bias-free neuron intervention.')
    return entries


def prepare(analysis_summary_path, word_cases_path, supplemental_cases_path, output, *,
            config=checkpoint.GPT2Config()):
    """Write plan.json and three immutable case exports, never checkpoint copies."""
    summary_record = _record(analysis_summary_path)
    summary = json.loads(Path(analysis_summary_path).read_text())
    if summary.get('format') != 'pluto-paired-analysis-v1' or summary.get('complete') is not True:
        raise ValueError('a verified completed paired-analysis summary is required')
    main, _, main_records = _case_input(word_cases_path)
    supplemental, _, supplemental_records = _case_input(supplemental_cases_path)
    if (main['format'] != 'pluto-paired-word-cases-v1'
            or supplemental['format'] != paired_supplemental_cases.FORMAT
            or supplemental['source_word_cases'] != main_records[0]
            or supplemental['source_word_packed_batch'] != main_records[1]
            or supplemental['manifest'] != main['manifest']):
        raise ValueError('main and supplemental cases must share the exact frozen provenance')
    records = [summary_record, *main_records, *supplemental_records,
               *summary['frozen_inputs'], *summary['terminal_records']]
    if (main_records[0] not in summary['frozen_inputs'] or main_records[1] not in summary['frozen_inputs']
            or main['manifest'] not in summary['frozen_inputs']):
        raise ValueError('trajectory did not score these exact main cases')
    _verify_records(records)
    manifest = json.loads(Path(main['manifest']['path']).read_text())
    root = Path(manifest['root']).resolve()
    state_path = root / 'state.json'
    if _record(state_path) not in summary['terminal_records']:
        raise ValueError('completed analysis lacks current terminal state provenance')
    state = json.loads(state_path.read_text())
    gate = summary.get('determinism_verification')
    if (manifest.get('seconds_per_arm') != 14400 or state.get('phase') != 'training_complete'
            or not gate or gate.get('evidence', {}).get('status') != 'verified'
            or state.get('determinism_gate') != gate['evidence']):
        raise ValueError('requires completed four-hour deterministic paired training')
    _verify_records(gate['records'])
    pairs, endpoints = _selected_pairs(summary)
    inventories = {}
    for arm in ARMS:
        inventory_path = root / arm / 'checkpoints.json'
        if _record(inventory_path) not in summary['terminal_records']:
            raise ValueError('completed analysis lacks checkpoint inventory provenance')
        inventory = json.loads(inventory_path.read_text())
        by_step = {item['step']: item for item in inventory}
        if len(by_step) != len(inventory) or 0 not in by_step:
            raise ValueError('invalid checkpoint inventory')
        inventories[arm] = by_step
        run = state['runs'][arm]
        if (run.get('returncode') != 0 or run.get('initial_weights_match') is not True
                or run.get('stop_reason') != 'time_limit'
                or not math.isfinite(run.get('training_elapsed_seconds', 0))
                or run.get('training_elapsed_seconds', 0) < 14400
                or run.get('final_step') != endpoints[arm]['step']
                or run.get('final_step') != max(by_step)
                or run.get('final_checkpoint') != endpoints[arm]['path']):
            raise ValueError('terminal training state disagrees with completed endpoints')
    common = sorted(set(inventories['original']) & set(inventories['replacement']))
    if common != summary['plan']['matched_steps']:
        raise ValueError('trajectory common steps disagree with actual completed inventories')
    for pair in [*pairs, endpoints]:
        for arm in ARMS:
            if pair[arm] != inventories[arm].get(pair[arm]['step']):
                raise ValueError('trajectory checkpoint differs from actual completed inventory')
    piece_ids = supplemental['selection']['union_piece_ids']
    declared_pieces = sorted({token for split in paired_word_cases.SPLITS
                             for occurrence in manifest['alignment'][split]['occurrences']
                             for arm in ARMS for token in occurrence[f'{arm}_ids']})
    if (piece_ids != declared_pieces or len(piece_ids) != 9
            or any(type(token) is not int or not 0 <= token < config.vocab_size for token in piece_ids)):
        raise ValueError('expected exactly the nine native union word-piece IDs')
    specs = checkpoint.tensor_manifest(config)
    verified = {}
    for item in summary['checkpoints']:
        if item['path'] in verified and verified[item['path']] != item['weight_sha256']:
            raise ValueError('conflicting completed checkpoint hash records')
        verified[item['path']] = item['weight_sha256']
    for pair in pairs:
        for arm in ARMS:
            _verify_checkpoint(pair[arm], verified, root, arm, specs)
    output = _new_output(output, (Path(analysis_summary_path).resolve().parent,
                                  Path(word_cases_path).resolve().parent,
                                  Path(supplemental_cases_path).resolve().parent,
                                  *(root / arm / 'checkpoints' for arm in ('initial', *ARMS))))
    subset_indices = {
        'word_next_native': [index for index, case in enumerate(supplemental['cases'])
                             if case['kind'] == 'word_next_native'],
        'shared_piece': [index for index, case in enumerate(supplemental['cases'])
                         if case['kind'] == 'shared_piece']}
    if any(not indices for indices in subset_indices.values()):
        raise ValueError('both supplemental word and shared-piece controls are required')
    interventions = [entry for pair in pairs for entry in _interventions(pair, specs, piece_ids)]
    output.mkdir()
    exports = {'main': export_cases(word_cases_path, output / 'main', rows_per_case=3)}
    for name, indices in subset_indices.items():
        exports[name] = export_cases(supplemental_cases_path, output / name, indices=indices,
                                     rows_per_case=4 if name == 'word_next_native' else 3)
    _verify_records(records + gate['records'])
    for pair in pairs:
        for arm in ARMS:
            _verify_checkpoint(pair[arm], verified, root, arm, specs)
    result = {'format': FORMAT, 'complete': True, 'patches_materialized': False,
              'completion_meaning': 'Immutable plan and case exports prepared; interventions have NOT run.',
              'analysis_summary': summary_record, 'frozen_inputs': records[1:],
              'config': asdict(config), 'selected_steps': [pair['original']['step'] for pair in pairs],
              'step_selection': 'Earliest and latest common positive optimizer step; deduplicated. '
                                'Unequal time-matched endpoints never substitute.',
              'endpoint_context': {arm: endpoints[arm] for arm in ARMS},
              'word_piece_ids': piece_ids, 'interventions': interventions, 'exports': exports,
              'implementation': [_record(module.__file__) for module in
                                 (checkpoint, paired_word_cases, paired_supplemental_cases)] + [_record(__file__)],
              'limitations': ['A donor transfer measures a functional contribution on frozen contexts, '
                              'not a unique physical storage location for the word.',
                              'Whole-branch transfers can disrupt co-adapted components. Output-only '
                              'transfers include biases and must be interpreted separately.',
                              'All checkpoint sizes/hashes must be revalidated again immediately before '
                              'materialization/execution; this plan does not reserve immutable storage.']}
    paired_word_cases._write_json(output / 'plan.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('analysis-summary', 'word-cases', 'supplemental-cases', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(argv)
    prepare(args.analysis_summary, args.word_cases, args.supplemental_cases, args.output)


if __name__ == '__main__':
    main()
