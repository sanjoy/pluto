"""Audit an outer embedding-row/complement factorial using saved native logits.

E replaces selected embedding rows in A. C replaces every OTHER tensor with
the donor D's bytes, retaining A's entire token embedding. EC combines E and
C, so D differs from EC only in the remaining embedding rows. These factors
are not the input/output roles of the inner native embedding-factorial probe.
No model execution, checkpoint mutation, or unique-storage fraction is implied.
"""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import tempfile

import numpy as np

from . import embedding_factorial_mechanism as mechanism
from . import embedding_factorial_readout as native
from .checkpoint import GPT2Checkpoint, GPT2Config, tensor_manifest


FORMAT = 'pluto-paired-outer-factorial-readout-v1'
CELLS = ('A', 'E', 'C', 'EC', 'D')
EFFECTS = ('E', 'C', 'interaction', 'joint')
METRICS = ('margin', 'normalizer', 'log_probability')
_FIELDS = {'margin': 'margin', 'normalizer': 'log_partition_relative_to_rival',
           'log_probability': 'log_probability'}
_require = mechanism.require


def decompose_logits(logits, target):
    """Return an A-anchored outer factorial and conditional D-minus-EC effect.

    All inputs are actual finite native FP32 rows. The same highest A
    non-target rival is retained for E, C, EC AND D. In particular, D is not
    assigned a new convenient rival when measuring the remaining-row effect.
    """
    _require(isinstance(logits, dict) and set(logits) == set(CELLS),
             'expected exactly A/E/C/EC/D logit rows')
    rows = {name: np.asarray(value) for name, value in logits.items()}
    first = rows['A']
    _require(first.ndim == 1 and first.size >= 2 and all(
        row.shape == first.shape and row.dtype == np.dtype('<f4')
        and np.isfinite(row).all() for row in rows.values()),
        'expected equally sized finite one-dimensional FP32 logit rows')
    inner = mechanism.decompose_logits(
        {'AA': rows['A'], 'JA': rows['E'], 'AJ': rows['C'], 'JJ': rows['EC']}, target)
    cells = {outer: dict(inner['cells'][cell]) for outer, cell in
             (('A', 'AA'), ('E', 'JA'), ('C', 'AJ'), ('EC', 'JJ'))}
    rival = inner['rival_id']
    values = rows['D'].astype(np.float64)
    maximum = float(values.max())
    log_mass = math.log(float(np.exp(values - maximum).sum(dtype=np.float64)))
    donor_target, donor_rival = float(values[target]), float(values[rival])
    cells['D'] = dict(target_logit=donor_target, rival_logit=donor_rival,
        margin=donor_target - donor_rival,
        log_partition_relative_to_rival=math.fsum((maximum - donor_rival, log_mass)),
        log_probability=-math.fsum((maximum - donor_target, log_mass)))
    for cell in cells.values():
        cell['probability'] = math.exp(cell['log_probability'])
    # Rename explicitly: these are whole-factor changes, NOT input/output roles.
    effects = {name: {outer: values[old] for outer, old in
                       (('E', 'input'), ('C', 'output'),
                        ('interaction', 'interaction'), ('joint', 'joint'))}
               for name, values in inner['effects'].items()}
    residual = {name: cells['D'][field] - cells['EC'][field]
                for name, field in _FIELDS.items()}
    error = abs(math.fsum((residual['margin'], -residual['normalizer'],
                          -residual['log_probability'])))
    return dict(target_id=target, rival_id=rival, cells=cells, effects=effects,
        remaining_embedding_residual=residual,
        anchored_logit_interaction=inner['anchored_logit_interaction'],
        max_effect_identity_error=max(error, inner['max_effect_identity_error']))


def _sequence(tokens):
    """Sum distinct teacher-forced positions, each with its own fixed rival."""
    cells = {cell: {field: math.fsum(token['cells'][cell][field] for token in tokens)
                    for field in _FIELDS.values()} for cell in CELLS}
    for cell in cells.values():
        cell['probability'] = math.exp(cell['log_probability'])
    return dict(cells=cells,
        effects={name: {kind: math.fsum(token['effects'][name][kind] for token in tokens)
                        for kind in EFFECTS} for name in METRICS},
        remaining_embedding_residual={name: math.fsum(
            token['remaining_embedding_residual'][name] for token in tokens)
            for name in METRICS})


def _identity(path):
    info = Path(path).stat()
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def _conditioned_checkpoint(base, conditioned, donor, records, config):
    """Verify C's construction from actual source/output bytes, not flags alone."""
    a, d = (Path(base['patch']['paths'][side]) for side in ('A', 'D'))
    c = Path(conditioned['patch']['paths']['A'])
    _require(c not in (a, d), 'conditioned checkpoint must be a distinct copy')
    _require(Path(conditioned['patch']['paths']['D']) == d,
             'conditioned donor does not match base donor')
    _require(Path(donor['patch']['paths']['A']) == d
             and Path(donor['patch']['paths']['D']) == a,
             'donor report must be the reciprocal base endpoints')
    for report in (conditioned, donor):
        _require(report['patch']['selected_rows'] == base['patch']['selected_rows']
                 and report['patch']['actual_changed_rows'] == base['patch']['actual_changed_rows'],
                 'selected/changed embedding rows differ across reports')
    patch_path = c / 'patch.json'
    patch_record = native._register(records, patch_path)
    patch = native._json(patch_path)
    specs = tensor_manifest(config)
    selected = sorted(spec.name for spec in specs if spec.index != 0)
    _require(patch.get('format') == 'pluto-paired-weight-patch-v1'
             and patch.get('complete') is True and patch.get('config') == asdict(config)
             and patch.get('operation') == 'exact donor byte replacement; no interpolation',
             'invalid conditioned checkpoint construction')
    _require(GPT2Config(**patch['config']) == config, 'invalid conditioned configuration types')
    _require(patch.get('selection') == dict(tensors=selected, embedding_rows=[],
        embedding_row_byte_ranges=[], embedding_is_tied_to_lm_head=True),
        'C must select exactly every non-token-embedding tensor')
    _require(all(patch.get('validation', {}).get(key) is True for key in (
        'all_weights_finite', 'sources_unchanged', 'selected_bytes_equal_replacement',
        'unselected_bytes_equal_original', 'no_hardlinks')),
        'missing conditioned construction validation')
    _require(patch['sources']['original']['path'] == str(a)
             and patch['sources']['replacement']['path'] == str(d)
             and patch['output']['path'] == str(c),
             'conditioned construction source/output paths do not match reports')
    allowed = {spec.filename for spec in specs}
    snapshots, checkpoints = {}, {}
    for name, path in (('A', a), ('D', d), ('C', c)):
        _require(path.is_dir() and not path.is_symlink()
                 and {p.name for p in path.iterdir()} <= allowed | {'patch.json'},
                 'unexpected conditioned checkpoint layout')
        declared = (patch['output']['weights_sha256'] if name == 'C' else
                    patch['sources']['original' if name == 'A' else 'replacement']['weights_sha256'])
        _require(set(declared) == allowed, 'incomplete conditioned construction hashes')
        checkpoints[name] = GPT2Checkpoint(path, config, check_finite=True)
        for spec in specs:
            item = native._register(records, path / spec.filename)
            _require(item['bytes'] == spec.nbytes and item['sha256'] == declared[spec.filename],
                     'conditioned construction weight hash mismatch')
            snapshots[item['path']] = _identity(item['path'])
    for spec in specs:
        source = 'A' if spec.index == 0 else 'D'
        _require(native._bits_equal(checkpoints['C'][spec.name], checkpoints[source][spec.name]),
                 'C bytes do not match exact embedding/complement selection')
        output_identity = snapshots[str(c / spec.filename)][:2]
        _require((c / spec.filename).stat().st_nlink == 1
                 and all(output_identity != snapshots[str(path / spec.filename)][:2]
                         for path in (a, d)), 'conditioned checkpoint shares a source inode/hardlink')
    implementation = patch.get('implementation', {})
    source_record = native._register(records, implementation['path'])
    _require(source_record['sha256'] == implementation['sha256'],
             'conditioned patch implementation changed')
    loader = native._register(records, Path(native.__file__).with_name('checkpoint.py'))
    _require(loader['sha256'] == implementation['checkpoint_loader_sha256'],
             'conditioned patch checkpoint loader changed')
    return dict(patch=patch_record, paths={'A': str(a), 'D': str(d), 'C': str(c)},
        selected_tensors=selected, exact_bytes_verified=True,
        no_source_hardlinks_verified=True, identities=snapshots)


def analyze(base_readout, conditioned_readout, donor_readout, output, config=GPT2Config()):
    """Authenticate three completed native reports; exclusively publish CPU math.

    Native reader re-execution is CPU-only and must reproduce each saved report
    exactly. Its execution ledgers are required, but are not independent process
    attestation. All input records and C's file identities are checked again
    after computation; missing or changed evidence prevents publication.
    """
    output = Path(output)
    _require(not output.exists() and not output.is_symlink(), 'output already exists')
    records, saved, readout_records = {}, {}, {}
    for name, path in (('base', base_readout), ('conditioned', conditioned_readout),
                       ('donor', donor_readout)):
        readout_records[name] = native._register(records, path)
        item = native._json(path)
        _require(item.get('format') == 'pluto-embedding-factorial-readout-v1'
                 and item.get('execution_provenance', {}).get('verified') is True,
                 'completed factorial report with execution provenance required')
        forbidden = [Path(p).resolve() for p in item['patch']['paths'].values()]
        forbidden.append(Path(item['native_metadata']['path']).resolve().parent)
        _require(all(root != output.resolve() and root not in output.resolve().parents
                     for root in forbidden),
                 'output must be outside source checkpoints and native score directories')
        for record in item['files']:
            native._register(records, record['path'], record)
        saved[name] = item
    # Explicit own-file inclusion also works when invoked as `python -m ...`.
    for path in (__file__, mechanism.__file__, native.__file__):
        native._register(records, path)
    with tempfile.TemporaryDirectory(prefix='pluto-outer-factorial-') as temporary:
        for name, item in saved.items():
            fresh = native.analyze(item['cases']['path'],
                Path(item['native_metadata']['path']).parent,
                item['patch']['patch']['path'], Path(temporary) / (name + '.json'),
                expected_rows=item['patch']['selected_rows'],
                case_kind=item['case_kind_selection'],
                execution_record=item['execution_provenance']['record']['path'], config=config)
            _require(fresh == item, 'saved readout disagrees with independent revalidation: ' + name)
    base, conditioned, donor = (saved[name] for name in ('base', 'conditioned', 'donor'))
    for other in (conditioned, donor):
        _require(other['cases'] == base['cases']
                 and other['case_kind_selection'] == base['case_kind_selection']
                 and len(other['per_case']) == len(base['per_case']),
                 'case source/suite/order mismatch')
    construction = _conditioned_checkpoint(base, conditioned, donor, records, config)
    # Only five mmap views are needed; no collection of complete copied logits.
    mapping = {'A': ('base', 'AA'), 'E': ('base', 'JJ'),
               'C': ('conditioned', 'AA'), 'EC': ('conditioned', 'JJ'), 'D': ('donor', 'AA')}
    arrays = {}
    for name, (report, cell) in mapping.items():
        path = Path(saved[report]['native_metadata']['path'])
        metadata = native._json(path)
        arrays[name] = np.memmap(path.parent / (cell + '.logits.f32.bin'),
            dtype='<f4', mode='r', shape=tuple(metadata['logits_shape']))
    _require(len({array.shape for array in arrays.values()}) == 1, 'native logit geometry mismatch')
    labels = ('native_case_index', 'source_case_index', 'kind', 'split', 'context_id',
              'prefix_domain', 'target', 'target_ids', 'spelling_variant', 'candidate_pair',
              'target_source_domain', 'piece_id', 'word_has_leading_space', 'predictions')
    per_case = []
    for index, item in enumerate(base['per_case']):
        _require(item['native_case_index'] == index, 'unexpected native case ordering')
        signature = {key: item[key] for key in labels if key in item}
        for other in (conditioned, donor):
            candidate = other['per_case'][index]
            _require({key: candidate[key] for key in labels if key in candidate} == signature,
                     'case identity/prefix/target/order mismatch')
        tokens = []
        for position, target in enumerate(item['target_ids']):
            token = decompose_logits({name: array[index, position] for name, array in arrays.items()}, target)
            for cell, (report, inner) in mapping.items():
                native._close(token['cells'][cell]['log_probability'], saved[report]['per_case'][index]
                    ['cells'][inner]['token_log_probability'][position], 'outer token log probability')
            exposure = item['predictions'][position]
            token.update(target_position=position,
                prediction_row=exposure['prediction_row'],
                causal_prefix_length=exposure['causal_prefix_length'],
                causal_prefix_sha256=exposure['causal_prefix_sha256'],
                selected_embedding_input_ids_in_prefix=exposure['patched_input_ids'],
                selected_embedding_rows_absent_from_prefix=exposure['input_path_must_be_unchanged'])
            tokens.append(token)
        result = {key: value for key, value in signature.items() if key != 'predictions'}
        result.update(tokens=tokens, selected_sequence=_sequence(tokens))
        if item['kind'] in ('word', 'word_next_native'):
            result.update(first_piece=_sequence(tokens[:1]), suffix=_sequence(tokens[1:3]),
                          word_three=_sequence(tokens[:3]))
            if item['kind'] == 'word_next_native':
                result['exact_next_native_token'] = _sequence(tokens[3:4])
        per_case.append(result)
    for record in records.values():
        _require(native._record(record['path']) == record, 'source changed during outer decomposition')
    for path, identity in construction['identities'].items():
        _require(_identity(path) == identity, 'checkpoint identity changed during outer decomposition')
    result = dict(format=FORMAT, complete=True, source_readouts=readout_records,
        execution_provenance={name: item['execution_provenance'] for name, item in saved.items()},
        conditioned_checkpoint=construction, per_case=per_case,
        files=[records[path] for path in sorted(records)],
        model_forward_performed=False, weights_edited=False, goal_completion_claimed=False,
        definitions=dict(
            cells='A=base AA; E=base JJ; C=conditioned AA; EC=conditioned JJ; D=reciprocal donor AA.',
            E='Replace selected token-embedding rows in A in both tied roles.',
            C='Replace all non-token-embedding tensors with donor D; retain the entire A embedding.',
            effects='E=E-A; C=C-A; interaction=EC-E-C+A; joint=EC-A. These are NOT input/output effects.',
            remaining_embedding_residual='D-EC: conditional effect of replacing remaining embedding rows after E and C; not a unique storage fraction.',
            rival='Highest A non-target logit, lower-ID tie break; fixed across A/E/C/EC/D per prediction.',
            decomposition='log_probability effect = fixed-rival margin effect - fixed-rival normalizer effect.',
            exposure='Selected-row input exposure refers only to the embedding change; absence does not imply C, E logits, or their output dictionaries are unchanged.',
            sequence='Sum teacher-forced per-position metrics; probability is exp(sum log probabilities). Each position has its own fixed rival.',
            weighting='Per-case aliases retained without aggregation; first/suffix/word-three and exact following token remain separate.'),
        limitations=[
            'Revalidates recorded native evidence; does not independently attest native process execution.',
            'Synthetic CPU fixtures exercise validation/math, not GPU forward correctness.',
            'C includes positions, normalization, and all transformer tensors, not only transformer blocks.',
            'D-minus-EC is conditional on the other factors; it is not the effect of remaining rows in A.',
            'Remaining embedding rows include physical padding, whose logits are masked by the native model.',
            'Fixed-rival normalizer changes can contain genuine competing-logit interactions, not merely softmax curvature.',
            'Teacher-forced candidate probabilities are not free generation, complete-string probability, or a unique word-storage attribution.'])
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base-readout', 'conditioned-readout', 'donor-readout', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(argv)
    analyze(args.base_readout, args.conditioned_readout, args.donor_readout, args.output)


if __name__ == '__main__':
    main()
