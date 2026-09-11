"""Read exact attention/MLP donor-transfer evidence, without executing a model.

The caller supplies six native paired_loss_probe directories: recipient, donor,
and patched models on the main and supplemental frozen suites. Every directory
must have a successful execution-time before/after hash record. Baselines may
use an independently verified, byte-identical copy-control checkpoint.

Scores are probabilities of specified teacher-forced token sequences. A shift
in Nuveth/Exeunt odds is NOT evidence that either word became more likely: both
absolute word probabilities, each subtoken, and unrelated/shared-piece controls
are retained. Whole-branch transfers include their pre-LayerNorm and biases;
output-write transfers include the projection bias, not just feature directions.
Neither intervention establishes a unique storage location or necessity.
"""

from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from .checkpoint import GPT2Checkpoint, GPT2Config, tensor_manifest
from .embedding_factorial_readout import _json, _record, _register, _require
from .paired_word_cases import _write_json
from . import paired_case_contract as case_contract


FORMAT = 'pluto-paired-branch-readout-v1'
ROLES = ('recipient', 'donor', 'patched')
SUITES = ('main', 'supplemental')
WORDS = ('Exeunt', 'Nuveth')
MIN_TRANSFER_GAP = 1e-6  # nats; tiny denominators do not support a useful ratio.


def effect(values):
    """Unclipped log-probability effect and a denominator-qualified ratio.

    Fractions outside [0, 1] are possible and are retained. A zero/near-zero
    donor gap gives null, not infinity, zero, or a claimed transfer percentage.
    This descriptive ratio is not a confidence interval or a causal mediation
    fraction; interactions and collateral degradation can be substantial.
    """
    _require(set(values) == set(ROLES) and all(math.isfinite(v) for v in values.values()),
             'effects require three finite native-derived log probabilities')
    gap = values['donor'] - values['recipient']
    change = values['patched'] - values['recipient']
    return {'donor_minus_recipient': gap, 'patched_minus_recipient': change,
            'patched_minus_donor': values['patched'] - values['donor'],
            'transfer_fraction': change / gap if abs(gap) > MIN_TRANSFER_GAP else None,
            'transfer_fraction_minimum_absolute_gap_nats': MIN_TRANSFER_GAP,
            'transfer_fraction_qualified': abs(gap) > MIN_TRANSFER_GAP}


def _weights(path, declared, records, config):
    path = Path(path)
    _require(path.is_dir() and not path.is_symlink(), 'checkpoint must be a real directory')
    specs = tensor_manifest(config)
    names = {spec.filename for spec in specs}
    _require(set(declared) == names and {p.name for p in path.iterdir()} <= names | {'patch.json'},
             'unexpected checkpoint layout/hash inventory')
    # Check finite values rather than trusting the producer's validation booleans.
    GPT2Checkpoint(path, config, check_finite=True)
    items = []
    for spec in specs:
        item = _register(records, path / spec.filename)
        _require(item['bytes'] == spec.nbytes and item['sha256'] == declared[spec.filename],
                 'checkpoint weight hash mismatch: ' + item['path'])
        items.append(item)
    return items


def _patch(path, records, config, *, expected=None):
    """Verify every actual selected donor/unselected recipient file and inode."""
    _register(records, path)
    patch = _json(path)
    _require(patch.get('format') == 'pluto-paired-weight-patch-v1'
             and patch.get('complete') is True and patch.get('config') == asdict(config),
             'invalid patch provenance/configuration')
    GPT2Config(**patch['config'])
    selection = patch['selection']
    selected = selection.get('tensors')
    _require(isinstance(selected, list) and selected == sorted(set(selected))
             and selection.get('embedding_rows') == []
             and selection.get('embedding_row_byte_ranges') == [],
             'branch/copy patch must use whole tensors and no embedding rows')
    specs = tensor_manifest(config)
    _require(set(selected) <= {spec.name for spec in specs}, 'unknown selected tensor')
    if expected is not None:
        _require(selected == sorted(expected), 'patch differs from declared tensor intervention')
    for flag in ('all_weights_finite', 'sources_unchanged', 'selected_bytes_equal_replacement',
                 'unselected_bytes_equal_original', 'no_hardlinks'):
        _require(patch.get('validation', {}).get(flag) is True, 'missing patch validation: ' + flag)
    sources = {role: patch['sources'][key] for role, key in
               (('recipient', 'original'), ('donor', 'replacement'))}
    paths = {role: Path(item['path']).resolve() for role, item in sources.items()}
    paths['patched'] = Path(patch['output']['path']).resolve()
    _require(Path(path).resolve() == paths['patched'] / 'patch.json', 'patch location mismatch')
    hashes = {role: item['weights_sha256'] for role, item in sources.items()}
    hashes['patched'] = patch['output']['weights_sha256']
    weight_records = {role: _weights(paths[role], hashes[role], records, config) for role in ROLES}
    for spec in specs:
        source = 'donor' if spec.name in selected else 'recipient'
        _require(hashes['patched'][spec.filename] == hashes[source][spec.filename],
                 'patch selected/unselected bytes do not match their actual source')
        output_info = (paths['patched'] / spec.filename).stat()
        for role in ('recipient', 'donor'):
            source_info = (paths[role] / spec.filename).stat()
            _require((output_info.st_dev, output_info.st_ino) !=
                     (source_info.st_dev, source_info.st_ino), 'checkpoint copy shares source inode')
    return {'paths': paths, 'hashes': hashes, 'weight_records': weight_records,
            'record': _record(path), 'selection': selected}


def _intervention(item, patch, config):
    kind = item.get('kind', '')
    match = re.fullmatch(r'(attention|mlp)_(whole_branch|output_write)', kind)
    _require(match is not None and item.get('embedding_rows') == [], 'not a branch intervention')
    branch, scope = match.groups()
    selected = item['tensors']
    blocks = {int(m.group(1)) for name in selected
              if (m := re.match(r'blocks\.(\d+)\.', name))}
    _require(len(blocks) == 1, 'intervention must select exactly one transformer block')
    block = blocks.pop()
    _require(0 <= block < config.n_layers, 'invalid transformer block')
    prefix = f'blocks.{block}.'
    projection, norm = ('attn', 'ln1') if branch == 'attention' else ('mlp', 'ln2')
    specs = tensor_manifest(config)
    expected = [spec for spec in specs if (
        spec.name.startswith(prefix + projection + '.output.') if scope == 'output_write'
        else spec.name.startswith(prefix + projection + '.') or spec.name.startswith(prefix + norm + '.'))]
    _require(len(expected) == (2 if scope == 'output_write' else 6)
             and sorted(selected) == sorted(spec.name for spec in expected)
             and len(selected) == len(set(selected)), 'incorrect whole-branch/output-write selection')
    canonical = lambda value: json.dumps(value, sort_keys=True)
    _require(canonical(item.get('selection_specs')) == canonical([spec.to_dict() for spec in expected]),
             'declared tensor specs differ from the architecture')
    _require(patch['selection'] == sorted(selected), 'actual patch differs from intervention')
    _require(type(item.get('step')) is int and item['step'] > 0
             and (item.get('recipient_arm'), item.get('donor_arm')) in
             (('original', 'replacement'), ('replacement', 'original')), 'invalid matched-step direction')
    for role in ('recipient', 'donor'):
        checkpoint = item[role + '_checkpoint']
        _require(checkpoint.get('step') == item['step']
                 and Path(checkpoint['path']).resolve() == patch['paths'][role]
                 and checkpoint['sha256'] == patch['hashes'][role], 'intervention source identity mismatch')
    return {'block': block, 'branch': branch, 'scope': scope,
            'includes_pre_layernorm': scope == 'whole_branch', 'includes_output_bias': True}


def _historical_binding_register(records, expected, binding):
    """Retain the logical identity, freshly checking the bound physical file."""
    _require(binding['original'] == expected, 'archive changed historical identity')
    physical = binding['physical']
    _register(records, physical['path'], physical)
    return expected


def _historical_register(records, archive, path, expected):
    """Verify an old identity at its preserved location, without relabeling it.

    Only callers explicitly auditing pre-relocation evidence supply an archive.
    The original case/ledger retains its old logical record; our new audit's
    file inventory contains the actual physical archive record and the pinned
    manifest that explains the mapping. Live execution never uses this mode.
    """
    _require(str(Path(path).absolute()) == expected['path'],
             'historical record path differs from requested identity')
    binding, = archive.verify([expected])
    _historical_binding_register(records, expected, binding)
    for record in (archive.manifest_record, archive.owning_request_record):
        _register(records, record['path'], record)
    return expected


def _cases(path, suite, records, config, *, source_archive=None):
    record = _register(records, path)
    plan = _json(path)
    index=0 if suite=='main' else 1
    def register(path, expected):
        if source_archive is None:
            return _register(records, path, expected)
        return _historical_register(records, source_archive, path, expected)
    amended=case_contract.validate_plan(plan,register,_json)
    _require(plan.get('format') in (case_contract.LEGACY[index],case_contract.AMENDED[index])
             and type(plan.get('case_count')) is int
             and plan['case_count'] > 0 and plan['case_count'] == len(plan['cases'])
             and plan.get('context_length') == config.context_length
             and plan.get('vocab_size') == config.vocab_size
             and plan.get('eos_token_id') == config.vocab_size - 1,
             'invalid frozen cases geometry/word identity')
    batch = _register(records, plan['packed_batch']['path'], plan['packed_batch'])
    count, length = plan['case_count'], config.context_length
    _require(batch['bytes'] == 8 * count * length, 'wrong packed byte size')
    packed = np.memmap(batch['path'], dtype='<i4', mode='r', shape=(2, count, length))
    _require(not np.any(packed < 0) and not np.any(packed >= config.vocab_size)
             and np.array_equal(packed[0, :, 1:], packed[1, :, :-1]), 'invalid packed teacher forcing')
    for index, case in enumerate(plan['cases']):
        allowed = ('word', 'control') if suite == 'main' else ('word_next_native', 'shared_piece')
        kind, ids = case.get('kind'), case['target_ids']
        prefix_length = case['prefix']['length']
        _require(kind in allowed and case.get('case_index') == index
                 and type(case['case_index']) is int and case.get('split') in ('training', 'test')
                 and type(prefix_length) is int and prefix_length > 0
                 and len(ids) == (4 if kind == 'word_next_native' else 3)
                 and all(type(i) is int and 0 <= i < config.vocab_size for i in ids),
                 'invalid frozen case identity/IDs')
        rows = list(range(prefix_length - 1, prefix_length - 1 + len(ids)))
        _require(case['scored_rows'] == rows and rows[-1] < length
                 and packed[1, index, rows].tolist() == ids, 'case target order/row mismatch')
        actual_prefix = packed[0, index, :prefix_length].tobytes()
        _require(hashlib.sha256(actual_prefix).hexdigest() == case['prefix']['token_ids_sha256'],
                 'prefix hash mismatch')
        case_contract.metadata(case,amended=amended)
        if kind in ('word', 'word_next_native'):
            pieces = [bytes.fromhex(p) for p in case['target_source']['native_piece_bytes_hex']]
            _require(case['prefix_domain'] in ('original', 'replacement')
                     and len(pieces) == len(ids) and b''.join(pieces).hex() == case['target_source']['bytes_hex']
                     and b''.join(pieces[:3]) in (case['target'].encode(), b' ' + case['target'].encode()),
                     'candidate token pieces do not spell the specified word')
            if kind == 'word_next_native':
                _require(case.get('word_token_count') == 3 and case['next_native_token']['id'] == ids[3]
                         and case['next_native_token']['bytes_hex'] == pieces[3].hex(),
                         'invalid exact following native token')
        else:
            _require(case['target'] == 'control_next_3' and case['prefix_domain'] == 'shared',
                     'invalid control identity')
            if kind == 'shared_piece':
                _require(case.get('piece_id') == ids[0], 'shared-piece ID mismatch')
    return {'record': record, 'plan': plan, 'packed': packed, 'batch': batch}


def _execution(path, metadata, directory, inputs, outputs, records, *, source_archive=None):
    record = _register(records, path)
    run = _json(path)
    _require(run.get('format') == 'pluto-paired-probe-execution-v1'
             and type(run.get('pid')) is int and run['pid'] > 0
             and type(run.get('returncode')) is int and run['returncode'] == 0,
             'missing/failed native execution evidence')
    times = [datetime.fromisoformat(run[key].replace('Z', '+00:00'))
             for key in ('started_utc', 'finished_utc')]
    _require(all(t.tzinfo is not None and t.utcoffset() == timezone.utc.utcoffset(t) for t in times)
             and times[0] <= times[1], 'execution timestamps must be ordered UTC')
    _require(isinstance(run['inputs_before'], list) and run['inputs_before'] == run['inputs_after'],
             'native execution input hashes changed')
    if source_archive is not None:
        _require(isinstance(run['outputs'], list), 'invalid execution output inventory')
        historical = run['inputs_before'] + run['outputs']
        # One audit pins the manifest/owner for the whole ledger. Keep every
        # ordered row: duplicate and required-membership checks use OLD IDs.
        bindings = source_archive.verify(historical)
        _require(isinstance(bindings, list) and len(bindings) == len(historical),
                 'archive changed historical inventory length')
        bindings = iter(bindings)
    for field, required in (('inputs_before', inputs), ('outputs', outputs)):
        supplied = set()
        for item in run[field]:
            actual = (_register(records, item['path'], item) if source_archive is None
                      else _historical_binding_register(records, item, next(bindings)))
            _require(actual['path'] not in supplied, 'duplicate execution evidence record')
            supplied.add(actual['path'])
        _require(set(required) <= supplied, 'native execution omits required ' + field)
    if source_archive is not None:
        for pinned in (source_archive.manifest_record, source_archive.owning_request_record):
            _register(records, pinned['path'], pinned)
    command = run['command']
    _require(isinstance(command, list) and command and all(type(s) is str for s in command)
             and Path(command[0]).resolve() == Path(metadata['binary_file']).resolve(),
             'execution binary mismatch')
    flags, index = {}, 1
    while index < len(command):
        part = command[index]
        _require(part.startswith('--'), 'unexpected positional execution argument')
        if '=' in part:
            key, value = part[2:].split('=', 1)
        else:
            _require(index + 1 < len(command), 'missing execution argument value')
            key, value = part[2:], command[index + 1]
            index += 1
        _require(key not in flags, 'duplicate execution argument')
        flags[key] = value
        index += 1
    _require(set(flags) <= {'checkpoint', 'batch', 'output_dir', 'batch_sequences'}, 'unexpected probe flags')
    for flag, expected in (('checkpoint', metadata['checkpoint_directory']),
                           ('batch', metadata['batch_file']), ('output_dir', directory)):
        _require(flag in flags and Path(flags[flag]).resolve() == Path(expected).resolve(),
                 'execution command path mismatch: ' + flag)
    _require(int(flags.get('batch_sequences', '1')) == metadata['batch_sequences'],
             'execution batching mismatch')
    return record


def _scores(directory, execution, cases, role, patch, records, config, *, source_archive=None):
    directory = Path(directory).resolve()
    outputs = [_register(records, directory / name) for name in
               ('metadata.json', 'losses.f32.bin', 'argmax.i32.bin')]
    metadata = _json(directory / 'metadata.json')
    count, length = cases['plan']['case_count'], config.context_length
    expected = {'kind': 'paired_loss_probe', 'complete': True, 'temperature': 1,
                'byte_order': 'little', 'loss_dtype': '<f4', 'argmax_dtype': '<i4',
                'loss_file': 'losses.f32.bin', 'argmax_file': 'argmax.i32.bin',
                'case_count': count, 'passage_count': count, 'context_length': length,
                'output_shape': [count, length], 'vocab_size': config.vocab_size,
                'padded_vocab_size': config.padded_vocab_size, 'layers': config.n_layers,
                'width': config.d_model, 'heads': config.n_heads,
                'head_dimension': config.head_dim, 'feed_forward_width': config.d_ff,
                'checkpoint_unique_weight_count': len(tensor_manifest(config)),
                'finite_losses': True, 'argmax_valid': True, 'optimizer_steps': 0,
                'backward_calls': 0, 'checkpoint_writes': 0}
    _require(all(metadata.get(key) == value for key, value in expected.items()),
             'incompatible/incomplete native loss metadata')
    _require(type(metadata.get('batch_sequences')) is int and metadata['batch_sequences'] == 1,
             'this matched causal readout requires single-sequence native scoring')
    _require(Path(metadata['batch_file']).resolve() == Path(cases['batch']['path'])
             and metadata.get('batch_bytes') == cases['batch']['bytes']
             and Path(metadata['output_directory']).resolve() == directory,
             'native score batch/output identity mismatch')
    binary = _register(records, metadata['binary_file'])
    native_checkpoint = Path(metadata['checkpoint_directory']).resolve()
    checkpoint_name = re.fullmatch(r'step_(\d+)', native_checkpoint.name)
    _require(checkpoint_name is not None and type(metadata.get('checkpoint_step')) is int
             and metadata['checkpoint_step'] == int(checkpoint_name.group(1)),
             'native checkpoint directory/step mismatch')
    if native_checkpoint == patch['paths'][role]:
        weights = patch['weight_records'][role]
    else:
        _require(role != 'patched', 'patched native checkpoint must be the declared intervention')
        copy = _patch(native_checkpoint / 'patch.json', records, config, expected=[])
        _require(copy['paths']['recipient'] == patch['paths'][role]
                 and copy['hashes']['patched'] == patch['hashes'][role],
                 'baseline checkpoint is not an exact copy of the specified model')
        weights = copy['weight_records']['patched']
    required = [item['path'] for item in weights] + [binary['path'], cases['record']['path'], cases['batch']['path']]
    if (native_checkpoint / 'patch.json').exists():
        required.append(_register(records, native_checkpoint / 'patch.json')['path'])
    execution_record = _execution(execution, metadata, directory, required,
                                  [item['path'] for item in outputs], records,
                                  source_archive=source_archive)
    _require(all(item['bytes'] == count * length * 4 for item in outputs[1:]), 'native score byte size mismatch')
    losses = np.fromfile(directory / 'losses.f32.bin', dtype='<f4').reshape(count, length)
    argmax = np.fromfile(directory / 'argmax.i32.bin', dtype='<i4').reshape(count, length)
    _require(np.isfinite(losses).all() and not np.any(losses < 0)
             and not np.any(argmax < 0) and not np.any(argmax >= config.vocab_size),
             'invalid native loss/argmax values')
    return {'losses': losses, 'argmax': argmax, 'execution': execution_record, 'outputs': outputs}


def _model_score(losses, argmax, targets):
    logp = [-float(nll) for nll in losses]
    total = math.fsum(logp)
    return {'token_log_probabilities': logp, 'token_probabilities': [math.exp(v) for v in logp],
            'sequence_log_probability': total, 'sequence_probability': math.exp(total),
            'argmax_ids': [int(i) for i in argmax],
            'argmax_matches': [int(i) == j for i, j in zip(argmax, targets)]}


def _metric(items, positions):
    positions = list(positions)
    # A fourth-token variant is a different four-token event but must not give
    # its identical three-token word (or first-token prediction) extra weight.
    # Each metric uses only the actual prefix through its final scored target.
    unique = {}
    for item in items:
        key = (item['split'], item['prefix_ids_sha256'], item['prefix_length'],
               tuple(item['target_ids'][:max(positions) + 1]))
        unique.setdefault(key, item)
    items = list(unique.values())
    values, probabilities = {}, {}
    for role in ROLES:
        logs = [math.fsum(item['models'][role]['token_log_probabilities'][p] for p in positions)
                for item in items]
        values[role] = math.fsum(logs) / len(logs)
        probabilities[role] = {'mean_log_probability': values[role],
                              'geometric_mean_probability': math.exp(values[role]),
                              'arithmetic_mean_probability': math.fsum(math.exp(v) for v in logs) / len(logs)}
    return {'case_count': len(items), 'models': probabilities, 'effect_nats': effect(values)}


def _dedup(items):
    unique = {}
    for item in items:
        key = item['actual_case_identity']
        if key in unique:
            _require(item['models'] == unique[key]['models'], 'identical causal cases have different native scores')
        else:
            unique[key] = item
    return list(unique.values())


def _groups(items):
    groups = defaultdict(list)
    for item in items:
        for domain in (item['prefix_domain'], 'deduplicated_all'):
            groups[(item['suite'], item['kind'], item['split'], domain, item['target'])].append(item)
    result = []
    for (suite, kind, split, domain, target), aliases in sorted(groups.items()):
        unique = _dedup(aliases)
        count = len(unique[0]['target_ids'])
        metrics = {'sequence': _metric(unique, range(count)),
                   'first_three': _metric(unique, range(3)),
                   'tokens': [_metric(unique, [p]) for p in range(count)]}
        if count == 4:
            metrics['following_native_token'] = metrics['tokens'][3]
        result.append({'suite': suite, 'kind': kind, 'split': split, 'prefix_domain': domain,
                       'spelling_variant':unique[0].get('spelling_variant'),
                       'target': target, 'case_count': len(unique), 'source_case_count': len(aliases),
                       'metrics': metrics})
    return result


def _shared_pieces(items):
    groups = defaultdict(list)
    for item in items:
        if item['kind'] == 'shared_piece':
            groups[(item['split'], item['piece_id'])].append(item)
    return groups


def _variant_groups(items):
    """Project word groups by native leading-space variant, without reweighting."""
    return [{**group, 'leading_space': leading}
            for leading in (False, True)
            for group in _groups([item for item in items
                                  if item['kind'] in ('word', 'word_next_native')
                                  and item['leading_space'] == leading])]


def _causal_consistency(suites, native):
    """A score/argmax depends on its causal prefix, never the future targets.

    Matching a target additionally makes its FP32 NLL identical. Argmax does not
    even depend on the target label, so the first Exeunt and Nuveth cases must
    agree on that prediction when their initial prefixes agree.
    """
    losses, winners = {}, {}
    for suite, data in suites.items():
        for index, case in enumerate(data['plan']['cases']):
            for row in case['scored_rows']:
                prefix = data['packed'][0, index, :row + 1].tobytes()
                target = int(data['packed'][1, index, row])
                for role in ROLES:
                    key = (case['split'], role, prefix)
                    winner = native[role][suite]['argmax'][index, row].tobytes()
                    nll = native[role][suite]['losses'][index, row].tobytes()
                    _require(winners.setdefault(key, winner) == winner,
                             'identical causal prefixes have inconsistent argmax')
                    _require(losses.setdefault((key, target), nll) == nll,
                             'identical causal prefix/targets have inconsistent losses')


def _odds(items):
    by_context = defaultdict(dict)
    for item in _dedup([item for item in items if item['kind'] == 'word']):
        key = (item['split'],item.get('spelling_variant','title'), item['prefix_ids_sha256'], item['prefix_length'], item['leading_space'])
        _require(item['target'] not in by_context[key], 'ambiguous candidate IDs in a matched prefix')
        by_context[key][item['target']] = item
    per_context = []
    for (split,variant, prefix, length, leading), pair in sorted(by_context.items()):
        candidates=case_contract.PAIRS[variant]
        original,replacement=(candidates[role] for role in ('original','replacement'))
        _require(set(pair) == set(candidates.values()), 'matched prefix is missing one candidate word')
        words = {word: {role: pair[word]['models'][role]['sequence_log_probability'] for role in ROLES}
                 for word in candidates.values()}
        odds = {role: words[replacement][role] - words[original][role] for role in ROLES}
        per_context.append({'split': split, 'prefix_ids_sha256': prefix, 'prefix_length': length,
                            'spelling_variant':variant,'candidate_pair':candidates,
                            'leading_space': leading, 'word_log_probabilities': words,
                            'log_odds_replacement_over_original': odds, 'effect_nats': effect(odds),
                            'both_words_less_likely_after_patch': all(
                                words[word]['patched'] < words[word]['recipient'] for word in candidates.values())})
        if variant=='title':
            per_context[-1]['log_odds_Nuveth_over_Exeunt']=odds  # Historical field, never mislabeled lowercase.
    groups, variant_groups = [], []
    for leading in (None, False, True):
        for split,variant in ((split,variant) for split in ('training','test') for variant in case_contract.PAIRS):
            rows = [item for item in per_context if item['split'] == split and item['spelling_variant']==variant
                    and (leading is None or item['leading_space'] == leading)]
            if rows:
                odds = {role: math.fsum(item['log_odds_replacement_over_original'][role] for item in rows) / len(rows)
                        for role in ROLES}
                group = {'split': split, 'spelling_variant':variant,'candidate_pair':case_contract.PAIRS[variant],
                         'case_count': len(rows), 'mean_log_odds_replacement_over_original': odds,
                         'effect_nats': effect(odds), 'both_words_degraded_case_count': sum(
                             item['both_words_less_likely_after_patch'] for item in rows)}
                if variant=='title': group['mean_log_odds_Nuveth_over_Exeunt']=odds
                if leading is None:
                    groups.append(group)
                else:
                    variant_groups.append({**group, 'leading_space': leading})
    return {'definition': 'Within each spelling variant, log P(replacement native three tokens) - log P(original native three tokens); title/lowercase are never pooled.',
            'per_context': per_context, 'groups': groups, 'variant_groups': variant_groups}


def analyze(intervention, patch_path, main_cases_path, supplemental_cases_path, scores, output, *,
            executions, config=GPT2Config()):
    """Exclusively write a CPU-verified branch causal report.

    ``scores`` and ``executions`` both map recipient/donor/patched to dictionaries
    with main/supplemental paths. All six execution records are mandatory. A
    runner should additionally bind the intervention itself to its frozen plan
    and copy-control scores to the completed trajectory; this function checks
    the supplied intervention's actual tensor/source semantics, not a plan name.
    """
    output = Path(output).absolute()
    _require(not output.exists() and not output.is_symlink(), 'output already exists')
    _require(set(scores) == set(ROLES) and set(executions) == set(ROLES)
             and all(set(scores[role]) == set(SUITES) and set(executions[role]) == set(SUITES)
                     for role in ROLES), 'need exactly all six native score/execution pairs')
    protected = [Path(patch_path).resolve().parent, Path(main_cases_path).resolve().parent,
                 Path(supplemental_cases_path).resolve().parent]
    protected += [Path(scores[role][suite]).resolve() for role in ROLES for suite in SUITES]
    protected += [Path(intervention[role + '_checkpoint']['path']).resolve()
                  for role in ('recipient', 'donor')]
    _require(not any(output.resolve() == directory or directory in output.resolve().parents
                     for directory in protected), 'readout must be outside input directories')
    records = {}
    _register(records, __file__)
    _register(records,case_contract.__file__)
    patch = _patch(patch_path, records, config, expected=intervention['tensors'])
    selection = _intervention(intervention, patch, config)
    suites = {suite: _cases(path, suite, records, config) for suite, path in
              (('main', main_cases_path), ('supplemental', supplemental_cases_path))}
    main, supplemental = (suites[s]['plan'] for s in SUITES)
    _require(supplemental.get('source_word_cases') == suites['main']['record']
             and supplemental.get('source_word_packed_batch') == suites['main']['batch']
             and case_contract.experiment_identity(supplemental)==case_contract.experiment_identity(main),
             'supplemental/main provenance mismatch')
    native = {role: {suite: _scores(scores[role][suite], executions[role][suite], suites[suite],
                                    role, patch, records, config) for suite in SUITES} for role in ROLES}
    _causal_consistency(suites, native)
    per_case, word_scores, supplemental_word_keys = [], {}, set()
    for suite in SUITES:
        for index, case in enumerate(suites[suite]['plan']['cases']):
            rows, ids = case['scored_rows'], case['target_ids']
            prefix_length = case['prefix']['length']
            prefix_hash = case['prefix']['token_ids_sha256']
            identity = hashlib.sha256(json.dumps([case['split'], case['kind'], prefix_length,
                                                  prefix_hash, ids]).encode()).hexdigest()
            models = {role: _model_score(native[role][suite]['losses'][index, rows],
                                         native[role][suite]['argmax'][index, rows], ids) for role in ROLES}
            item = {'suite': suite, 'case_index': index, 'kind': case['kind'], 'split': case['split'],
                    'context_id': case['context_id'], 'prefix_domain': case['prefix_domain'],
                    'target': case['target'], 'target_ids': ids, 'prefix_length': prefix_length,
                    'prefix_ids_sha256': prefix_hash, 'actual_case_identity': identity,
                    'models': models, 'token_effects_nats': [effect({role: models[role]['token_log_probabilities'][p]
                                                                 for role in ROLES}) for p in range(len(ids))],
                    'sequence_effect_nats': effect({role: models[role]['sequence_log_probability'] for role in ROLES})}
            item.update(case_contract.metadata(case,amended=suites[suite]['plan']['format'] in case_contract.AMENDED))
            if case['kind'] in ('word', 'word_next_native'):
                item['leading_space'] = bytes.fromhex(case['target_source']['bytes_hex']).startswith(b' ')
                key = (case['split'], prefix_length, prefix_hash, tuple(ids[:3]))
                values = {role: (native[role][suite]['losses'][index, rows[:3]].tobytes(),
                                 native[role][suite]['argmax'][index, rows[:3]].tobytes()) for role in ROLES}
                if suite == 'main':
                    if key in word_scores:
                        _require(values == word_scores[key], 'duplicate word cases disagree')
                    word_scores[key] = values
                else:
                    _require(key in word_scores and word_scores[key] == values,
                             'supplemental first-three scores differ from main causal prefix')
                    supplemental_word_keys.add(key)
            if case['kind'] == 'shared_piece':
                item['piece_id'] = case['piece_id']
            per_case.append(item)
    _require(supplemental_word_keys == set(word_scores),
             'supplemental suite omits main-suite word contexts/candidates')
    _dedup(per_case)
    result = {'format': FORMAT, 'complete': True, 'goal_completion_claimed': False,
              'intervention': intervention, 'selection_semantics': selection,
              'patch': patch['record'], 'groups': _groups(per_case),
              'variant_groups': _variant_groups(per_case), 'odds': _odds(per_case),
              'per_case': per_case,
              'shared_piece_coverage': supplemental.get('selection', {}).get('coverage', {}),
              'shared_piece_groups': [
                  {'split': split, 'piece_id': piece,
                   'metrics': {'first_piece': _metric(rows, [0]), 'next_three': _metric(rows, range(3))}}
                  for (split, piece), rows in sorted(_shared_pieces(per_case).items())],
              'checks': {'actual_checkpoint_hashes_verified': True,
                         'exact_selected_donor_and_unselected_recipient_files': True,
                         'all_six_execution_before_after_hash_records_verified': True,
                         'main_supplemental_word_scores_byte_equivalent': True,
                         'all_scored_causal_prefix_argmax_and_target_nll_invariants': True,
                         'duplicate_actual_case_scores_equal': True},
              'aggregation': 'Equal weight per distinct split/kind/prefix-token-bytes/target-ID event; '
                             'domain aliases and repeated corpus occurrences are deduplicated. '
                             'Training and test are never pooled. Exact-next-token groups include '
                             'that fourth token; main word groups contain only three tokens.',
              'limitations': ['Whole-branch/output-write transplantation tests contribution, not unique storage or necessity.',
                              'Output-write includes weight and bias; whole branch also includes pre-LayerNorm.',
                              'A preference shift can coexist with both candidate words becoming less likely.',
                              'Unclipped transfer ratios are descriptive only; nontrivial donor gaps are required.',
                              'Native FP32 full-vocabulary losses are read, not independently recomputed from logits.',
                              'Teacher-forced native token sequences, not free-running success or sums over tokenizations.',
                              'Only the first pieces of different words share a prefix: later pieces condition on '
                              'different teacher-forced word prefixes. Their probability differences are not same-context logit margins.',
                              'The fourth target is the exact following native token, not any possible word boundary.',
                              'Interpret training discoveries separately from held-out test confirmation.'],
              'provenance': list(records.values()), 'implementation': _record(__file__)}
    for item in records.values():
        _require(_record(item['path']) == item, 'input changed during branch readout')
    _write_json(output, result)
    return result
