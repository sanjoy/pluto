"""Fixed coordinate-defined localization of the jointly causal complement.

E always replaces the selected embedding rows in BOTH tied roles. Q contains
positions/final LN, H the early half of whole blocks, L the late half. All
eight E-fixed cells retain recipient values in every unselected embedding row.
The explicit early_branches layout separates early attention with its pre-LN
(A), early MLPs with their pre-LN (M), and all remaining tensors (R). These
branches are interleaved residual computations, not independent modules.
This module prepares/checks exact patches and audits recorded native losses;
it never treats a coordinate-defined intervention as a unique storage site.

The preparation API fails closed until the outer controller actually exits
with complete authenticated evidence and an idle GPU. No native executable is
launched by this module: an owning runner must create execution ledgers before
the native-output/readout APIs can accept results.
"""

from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import embedding_factorial_readout as native
from . import paired_branch_readout as branch
from . import paired_outer_factorial as outer
from . import paired_weight_patch as patcher
from .checkpoint import GPT2Checkpoint, GPT2Config, tensor_manifest


FORMAT = 'pluto-paired-complement-localization-v1'
EARLY_BRANCHES_FORMAT = 'pluto-paired-early-branches-localization-v1'
training = outer.training
FP32_ABSOLUTE_TOLERANCE = 5e-5
FP32_RELATIVE_TOLERANCE = 5e-6
_require = native._require


def _axes(layout):
    """Keep the two fixed experiments explicit; never mutate module globals."""
    _require(type(layout) is str and layout in ('complement', 'early_branches'),
             'unknown localization layout')
    return ('Q', 'H', 'L') if layout == 'complement' else ('A', 'M', 'R')


def groups(config=GPT2Config(), *, layout='complement'):
    """Partition nonembedding tensors; small even layouts are CPU fixtures.

    A includes LN1 and both attention projections, including every bias. M
    includes LN2 and both MLP projections. Residual additions own no weights.
    R includes upstream positions, late blocks, and final LN: it is a
    complement, not a purely downstream computation.
    """
    _axes(layout)
    _require(config.n_layers >= 2 and config.n_layers % 2 == 0,
             'localization requires an even block count of at least two')
    specs = tensor_manifest(config)
    if layout == 'complement':
        result = {'Q': ['position_embedding.weight', 'final_norm.scale', 'final_norm.bias']}
        for group, blocks in (('H', range(config.n_layers // 2)),
                              ('L', range(config.n_layers // 2, config.n_layers))):
            result[group] = [s.name for s in specs if any(
                s.name.startswith(f'blocks.{block}.') for block in blocks)]
    else:
        result = {}
        for group, branches in (('A', ('ln1', 'attn')), ('M', ('ln2', 'mlp'))):
            result[group] = [s.name for s in specs if any(
                s.name.startswith(f'blocks.{block}.{branch}.')
                for block in range(config.n_layers // 2) for branch in branches)]
        early = set(result['A']) | set(result['M'])
        result['R'] = [s.name for s in specs[1:] if s.name not in early]
    names = [name for values in result.values() for name in values]
    _require(len(names) == len(set(names)) and set(names) == {s.name for s in specs[1:]},
             'groups must partition every non-token-embedding tensor')
    return result


def cube_subsets(*, layout='complement'):
    first, second, third = _axes(layout)
    return {'E': (), first: (first,), second: (second,), third: (third,),
            first+second: (first, second), first+third: (first, third),
            second+third: (second, third), 'EC': (first, second, third)}


def score_cube(values, *, layout='complement'):
    """Signed log-score contrasts in recipient and donor-complement contexts."""
    axes = _axes(layout)
    subsets = cube_subsets(layout=layout)
    _require(isinstance(values, dict) and set(values) == set(subsets)
             and all(type(v) in (int, float) and math.isfinite(v) for v in values.values()),
             'cube requires exactly eight finite nonboolean scores')
    by_set = {frozenset(subset): float(values[name]) for name, subset in subsets.items()}
    full = frozenset(axes)
    additions = {g: values[g] - values['E'] for g in axes}
    removals = {g: values['EC'] - by_set[full - {g}] for g in axes}
    pairs = {}
    for i, j in ((0, 1), (0, 2), (1, 2)):
        pair = (axes[i], axes[j])
        label = ''.join(pair)
        def interaction(context):
            a, b = (frozenset((g,)) for g in pair)
            return math.fsum((by_set[context | a | b], -by_set[context | a],
                              -by_set[context | b], by_set[context]))
        pairs[label] = dict(absent=interaction(frozenset()), present=interaction(full-set(pair)))
    return dict(additions=additions, removals=removals, pair_interactions=pairs,
                three_way=pairs[axes[0]+axes[1]]['present']-pairs[axes[0]+axes[1]]['absent'])


def _selection(subset, rows, config, *, layout='complement'):
    available = groups(config, layout=layout)
    _require(isinstance(subset, (tuple, list)) and all(type(g) is str and g in available for g in subset)
             and len(set(subset)) == len(subset), 'unknown or repeated group')
    _require(isinstance(rows, (tuple, list)) and all(type(r) is int and 0 <= r < config.vocab_size for r in rows)
             and len(set(rows)) == len(rows) and rows, 'invalid selected logical embedding rows')
    return sorted(name for group in subset for name in available[group]), sorted(rows)


def validate_model(patch_path, a, d, subset, rows, config=GPT2Config(), *, layout='complement'):
    """Authenticate every source/copy byte, including unselected/padded rows."""
    tensors, rows = _selection(subset, rows, config, layout=layout)
    a, d, patch_path = Path(a).absolute(), Path(d).absolute(), Path(patch_path).absolute()
    records = {}
    patch_record = native._register(records, patch_path)
    patch = native._json(patch_path)
    _require(patch.get('format') == 'pluto-paired-weight-patch-v1'
             and patch.get('complete') is True and patch.get('config') == asdict(config),
             'incomplete or wrong patch/configuration')
    GPT2Config(**patch['config'])
    width = config.d_model * 4
    _require(patch['selection'] == dict(tensors=tensors, embedding_rows=rows,
        embedding_row_byte_ranges=[[r*width, (r+1)*width] for r in rows],
        embedding_is_tied_to_lm_head=True), 'patch selection differs from fixed cube cell')
    _require(patch['sources']['original']['path'] == str(a)
             and patch['sources']['replacement']['path'] == str(d)
             and patch['output']['path'] == str(patch_path.parent), 'patch source/output binding differs')
    _require(all(patch['validation'].get(flag) is True for flag in (
        'all_weights_finite', 'sources_unchanged', 'selected_bytes_equal_replacement',
        'unselected_bytes_equal_original', 'no_hardlinks')), 'incomplete patch validation')
    paths = dict(recipient=a, donor=d, patched=patch_path.parent)
    hashes = dict(recipient=patch['sources']['original']['weights_sha256'],
                  donor=patch['sources']['replacement']['weights_sha256'], patched=patch['output']['weights_sha256'])
    weights = {role: branch._weights(path, hashes[role], records, config) for role, path in paths.items()}
    checkpoints = {role: GPT2Checkpoint(path, config, check_finite=True) for role, path in paths.items()}
    for spec in tensor_manifest(config):
        actual = checkpoints['patched'][spec.name]
        if spec.index == 0:
            previous = 0
            for row in rows:
                _require(native._bits_equal(actual[previous:row], checkpoints['recipient'][spec.name][previous:row])
                         and native._bits_equal(actual[row], checkpoints['donor'][spec.name][row]),
                         'selected/unselected embedding bytes differ')
                previous = row + 1
            _require(native._bits_equal(actual[previous:], checkpoints['recipient'][spec.name][previous:]),
                     'remaining/padded embedding bytes differ')
        else:
            role = 'donor' if spec.name in tensors else 'recipient'
            _require(native._bits_equal(actual, checkpoints[role][spec.name]), 'selected/unselected tensor bytes differ')
        info = (paths['patched']/spec.filename).stat()
        _require(info.st_nlink == 1 and all((info.st_dev, info.st_ino) !=
            ((source/spec.filename).stat().st_dev, (source/spec.filename).stat().st_ino)
            for source in (a, d)), 'patch is not an independent copy')
    implementation = patch['implementation']
    record = native._register(records, implementation['path'])
    _require(record['sha256'] == implementation['sha256'], 'patch implementation changed')
    loader = native._register(records, Path(native.__file__).with_name('checkpoint.py'))
    _require(loader['sha256'] == implementation['checkpoint_loader_sha256'], 'patch loader changed')
    for record in records.values():
        _require(native._record(record['path']) == record, 'source changed during model validation')
    return dict(paths={k: str(v) for k, v in paths.items()}, hashes=hashes, weight_records=weights,
        patch=patch_record, subset=list(subset), selection=tensors,
        records=[records[p] for p in sorted(records)])


def build_model(a, d, output, subset, rows, config=GPT2Config(), *, layout='complement'):
    tensors, rows = _selection(subset, rows, config, layout=layout)
    patcher.create_patch(a, d, output, tensors=tensors, embedding_rows=rows, config=config)
    return validate_model(Path(output)/'patch.json', a, d, subset, rows,
                          config=config, layout=layout)


def load_native(model, cases_path, suite, directory, execution, config=GPT2Config(), *,
                case_archive=None, execution_archive=None):
    """Require recorded success and validate ALL native context-token outputs.

    This deliberately bypasses only the old single-branch selection classifier,
    not its case, checkpoint, command, finite-value or execution-ledger checks.
    The model must first pass validate_model (or equivalent frozen source-map
    validation for a preserved E/EC reference checkpoint).
    """
    _require(suite in ('main', 'supplemental'), 'unknown native suite')
    records = {}
    for record in model['records']:
        native._register(records, record['path'], record)
    directory = Path(directory)
    _require(directory.is_dir() and not directory.is_symlink()
             and {p.name for p in directory.iterdir()} == {'metadata.json', 'losses.f32.bin', 'argmax.i32.bin'},
             'unexpected native output inventory')
    # Historical logical source paths are resolved only by the explicit,
    # authenticated archive supplied by the caller. New execution ledgers
    # normally use actual current paths and need no execution archive.
    for archive in (case_archive, execution_archive):
        if archive is not None:
            native._register(records, archive.manifest_record['path'], archive.manifest_record)
    case_options = {} if case_archive is None else {'source_archive': case_archive}
    cases = branch._cases(cases_path, suite, records, config, **case_options)
    adapter = dict(paths={k: Path(v) for k, v in model['paths'].items()},
                   hashes=model['hashes'], weight_records=model['weight_records'])
    execution_options = {} if execution_archive is None else {'source_archive': execution_archive}
    scored = branch._scores(directory, execution, cases, 'patched', adapter, records, config,
                            **execution_options)
    # The required before/after native records include the declared patch file.
    # An orphaned dump without a successful process ledger cannot pass here.
    seen, winners = {}, {}
    for case in cases['plan']['cases']:
        i = case['case_index']
        for row, target in zip(case['scored_rows'], case['target_ids']):
            prefix = cases['packed'][0, i, :row+1].tobytes()
            key = (prefix, target)
            value = (scored['losses'][i, row].tobytes(), int(scored['argmax'][i, row]))
            _require(seen.setdefault(key, value) == value, 'identical causal target has different native scores')
            _require(winners.setdefault(prefix, value[1]) == value[1],
                     'identical causal prefix has different argmax across targets')
    for record in records.values():
        _require(native._record(record['path']) == record, 'source changed during native validation')
    return dict(cases=cases, **scored, records=[records[p] for p in sorted(records)])


def compare_fp64_reference(measured, report, cell):
    """Bound FP32-loss versus FP64-logit scoring differences; never byte parity.

    GPU loss uses FP32 reductions/logarithms; the factorial reader uses FP64
    log-sum-exp on saved FP32 logits. A conservative 5e-5 absolute + 5e-6
    relative per-token tolerance accommodates rounding in these DIFFERENT
    computations. This is an explicit acceptance bound, not a rigorous error
    theorem; measured maxima are retained and near-bound effects need caution.
    """
    _require(cell in native.CELLS, 'unknown reference cell')
    plan = measured['cases']['plan']; maximum = 0.0; checked = 0
    kind = report.get('case_kind_selection')
    _require(report.get('format') == 'pluto-embedding-factorial-readout-v1'
             and report.get('cases') == measured['cases']['record']
             and kind in (None, 'word', 'control', 'word_next_native', 'shared_piece'),
             'FP64 reference case source/suite differs')
    indices = [c['case_index'] for c in plan['cases'] if kind is None or c['kind'] == kind]
    _require(indices and [item.get('source_case_index') for item in report['per_case']] == indices,
             'FP64 reference coverage/order differs')
    for item in report['per_case']:
        i = item['source_case_index']; case = plan['cases'][i]
        _require(all(item[k] == case[k] for k in ('kind', 'split', 'context_id', 'prefix_domain', 'target', 'target_ids')),
                 'FP64 reference case/target mismatch')
        for p, row in enumerate(case['scored_rows']):
            prefix = measured['cases']['packed'][0, i, :row+1].tobytes()
            prediction = item['predictions'][p]
            _require(prediction['prediction_row'] == row and prediction['causal_prefix_length'] == row+1
                     and prediction['causal_prefix_sha256'] == hashlib.sha256(prefix).hexdigest(),
                     'FP64 reference causal prefix mismatch')
            actual = float(measured['losses'][i, row]); expected = item['cells'][cell]['token_nll'][p]
            error = abs(actual-expected)
            _require(math.isfinite(expected) and error <= FP32_ABSOLUTE_TOLERANCE + FP32_RELATIVE_TOLERANCE*abs(expected),
                     'FP32 native loss differs from FP64 logit reference')
            _require(int(measured['argmax'][i, row]) == item['cells'][cell]['argmax_ids'][p],
                     'native argmax differs from reference')
            maximum = max(maximum, error); checked += 1
    return dict(checked_predictions=checked, maximum_absolute_nll_error=maximum,
                absolute_tolerance=FP32_ABSOLUTE_TOLERANCE, relative_tolerance=FP32_RELATIVE_TOLERANCE,
                exact_byte_equality_claimed=False)


def assert_native_copy_parity(reference, copied):
    """Same native kernel + same model bytes: every loss/argmax bit must agree."""
    _require(reference['cases']['record'] == copied['cases']['record'], 'copy-control case sources differ')
    for key in ('losses', 'argmax'):
        _require(native._bits_equal(reference[key], copied[key]), 'native all-context copy control differs')
    return dict(all_context_loss_and_argmax_byte_equal=True,
                predictions=int(reference['losses'].size))


def validate_outer_handoff(root, outer_directory):
    """Read-only, fail-closed handoff; never waits, signals, or launches a job."""
    root = Path(root).resolve(strict=True)
    directory = Path(outer_directory).absolute()
    _require(directory.parent == root and directory.is_dir() and not directory.is_symlink(),
             'outer directory must be a real immediate child of the amendment')
    _require(not (directory/'failure.json').exists(), 'outer controller recorded failure')
    request = native._json(directory/'request.json')
    _require(request.get('format') == outer.FORMAT and request.get('amendment_root') == str(root)
             and request.get('fixed_step') == 331 and request.get('training_restarted') is False,
             'wrong outer request identity')
    identity = request['runner_identity']; argv = identity['argv']
    _require(type(identity['pid']) is int and identity['pid'] > 0
             and type(identity['start_ticks']) is int and identity['start_ticks'] > 0
             and argv.count('-m') == 1
             and argv[argv.index('-m')+1] == 'weight_analysis.paired_outer_factorial'
             and argv.count('--root') == argv.count('--output') == 1
             and argv[argv.index('--root')+1] == str(root)
             and argv[argv.index('--output')+1] == str(directory), 'outer runner argv/identity mismatch')
    _require(not training.process_live(identity), 'outer runner is still live; no handoff')
    summary = native._json(directory/'summary.json')
    state = native._json(directory/'state.json')
    expected = {direction+'/'+suite for direction in
                ('replacement_to_original', 'original_to_replacement') for suite in outer.SUITES}
    _require(summary.get('format') == outer.FORMAT and summary.get('complete') is True
             and summary.get('amendment_root') == str(root) and summary.get('fixed_step') == 331
             and summary.get('training_restarted') is False and summary.get('goal_completion_claimed') is False
             and len(summary['completed']) == 6 and set(summary['completed']) == expected
             and state.get('format') == outer.FORMAT and state.get('phase') == 'complete'
             and state.get('runner_pid') == identity['pid'] and state.get('completed') == summary['completed'],
             'outer completion/state is incomplete or inconsistent')
    records = {}
    for record in [*request['frozen_inputs'], *summary['frozen_inputs'], *summary['artifacts']]:
        native._register(records, record['path'], record)
    declared = summary['artifacts']
    actual = [p for p in directory.rglob('*') if p.is_file()]
    _require(not any(p.is_symlink() for p in directory.rglob('*'))
             and len({r['path'] for r in declared}) == len(declared)
             and {str(p) for p in actual if p != directory/'summary.json'} == {r['path'] for r in declared},
             'outer completed output inventory differs')
    _require(all(r in summary['frozen_inputs'] for r in request['frozen_inputs'])
             and native._record(directory/'request.json') in summary['frozen_inputs'],
             'outer summary does not bind request/frozen inputs')
    plan = native._json(directory/'plan/plan.json')
    items = outer.select_endpoints(plan)
    _require(items == request['directions'], 'outer directions differ from fixed endpoint plan')
    for item in items:
        for role in ('recipient', 'donor'):
            endpoint = item[role+'_checkpoint']
            _require(endpoint['step'] == 331 and Path(endpoint['path']).name == 'step_331',
                     'localization endpoints must both be actual step 331')
            branch._weights(endpoint['path'], endpoint['sha256'], records, GPT2Config())
    original_request = native._json(root/'request.json')
    root_record = native._register(records, root/'request.json')
    _require(root_record in summary['frozen_inputs'], 'outer evidence did not freeze amended training request')
    runtime = request['runtime']
    _require(runtime.get('complete') is True and runtime.get('runtime_dlopen_covered') is False
             and set(runtime.get('environment', {})) == {'LD_LIBRARY_PATH'}, 'incomplete frozen runtime')
    for record in [*runtime['frozen_records'], *request['binaries'].values()]:
        _require(record in request['frozen_inputs'], 'outer request omitted runtime/binary binding')
        native._register(records, record['path'], record)
    for path in (directory/'summary.json', directory/'request.json', directory/'state.json',
                 directory/'plan/plan.json'):
        native._register(records, path)
    # Reobserve actual exit after the complete CPU hash/structure audit.
    _require(not training.process_live(identity), 'outer runner became live during handoff')
    training.require_idle_gpu(original_request['gpu'])
    return dict(request=request, summary=summary, plan=plan, items=items,
                gpu=original_request['gpu'], records=[records[p] for p in sorted(records)])


def prepare_plan(root, outer_directory, output_directory):
    """Publish only a NEW fixed manifest; no model copies or native launches."""
    root = Path(root).resolve(strict=True); output = Path(output_directory).absolute()
    _require(output.parent == root and not output.exists() and not output.is_symlink(),
             'plan output must be a NEW immediate child of the amendment')
    handoff = validate_outer_handoff(root, outer_directory)
    directory = Path(outer_directory).absolute(); config = GPT2Config()
    records = {r['path']: r for r in handoff['records']}
    cases = {}
    for suite, folder in (('main', 'word_cases'), ('supplemental', 'supplemental_cases')):
        checked = branch._cases(root/folder/'cases.json', suite, records, config)
        _require(checked['plan']['case_count'] == (188 if suite == 'main' else 265),
                 'unexpected prescribed main/supplemental case count')
        cases[suite] = checked['record']
    directions = {}
    for item in handoff['items']:
        name = item['donor_arm']+'_to_'+item['recipient_arm']
        references = {'E': {}, 'EC': {}}
        reference_checkpoints = {}
        for suite in outer.SUITES:
            for cell, path in (
                ('E', directory/'baseline_revalidation'/(item['name']+'_'+suite+'.json')),
                ('EC', directory/name/('outer_'+suite)/'readout.json')):
                record = native._register(records, path)
                report = native._json(path)
                _require(report['execution_provenance'].get('verified') is True
                         and report['patch']['selected_rows'] == item['embedding_rows'],
                         'reference readout lacks verified selected-row execution')
                _require(report['cases'] == cases['main' if suite == 'main' else 'supplemental'],
                         'reference suite case source differs')
                for frozen in report['files']:
                    native._register(records, frozen['path'], frozen)
                checkpoint = report['patch']['paths']['J']
                _require(reference_checkpoints.setdefault(cell, checkpoint) == checkpoint,
                         'reference checkpoint changes between suites')
                references[cell][suite] = record
        directions[name] = dict(recipient=item['recipient_checkpoint'], donor=item['donor_checkpoint'],
            rows=item['embedding_rows'], reference_checkpoints=reference_checkpoints,
            references=references)
    for module in (__file__, native.__file__, branch.__file__, outer.__file__, patcher.__file__):
        native._register(records, module)
    result = dict(format=FORMAT, phase='planned', config=asdict(config), step=331,
        amendment_root=str(root), outer_directory=str(directory),
        outer_summary=native._record(directory/'summary.json'),
        outer_runner=handoff['request']['runner_identity'], groups=groups(config),
        cells={name: list(subset) for name, subset in cube_subsets().items()},
        directions=directions, cases=cases, gpu=handoff['gpu'],
        binaries=handoff['request']['binaries'], runtime=handoff['request']['runtime'],
        loss_tolerance=dict(absolute=FP32_ABSOLUTE_TOLERANCE, relative=FP32_RELATIVE_TOLERANCE,
            applies_to='Selected native FP32 losses versus saved FP64-logit scores, per token; never silently relaxed'),
        control_protocol='For E and EC: preserved reference vs independent cube copy; identical full native losses/argmax, plus bounded selected-score FP64 comparison.',
        planned_native_scores='8 cube cells plus preserved E/EC references, each main+supplemental, in both endpoint directions.',
        interpretation='Coordinate-defined interventions with selected E rows tied in both roles; not unique storage.',
        native_execution_performed=False, runner_implemented=False,
        frozen_inputs=[records[p] for p in sorted(records)])
    training.verify_records(result['frozen_inputs'])
    _require(not training.process_live(result['outer_runner']), 'outer runner became live before plan publication')
    training.require_idle_gpu(result['gpu'])
    output.mkdir()
    training.publish(output/'plan.json', result)
    return result


def _cross_suite(main, supplemental):
    _require(supplemental['cases']['plan']['source_word_cases'] == main['cases']['record']
             and supplemental['cases']['plan']['source_word_packed_batch'] == main['cases']['batch'],
             'supplemental source binding differs')
    main_cases = main['cases']['plan']['cases']
    seen = {}
    for case in main_cases:
        if case['kind'] != 'word':
            continue
        key = (case['split'], case['prefix']['token_ids_sha256'], case['prefix']['length'], tuple(case['target_ids']))
        seen[key] = case
    count = 0
    for case in supplemental['cases']['plan']['cases']:
        if case['kind'] != 'word_next_native':
            continue
        key = (case['split'], case['prefix']['token_ids_sha256'], case['prefix']['length'], tuple(case['target_ids'][:3]))
        _require(key in seen, 'supplemental word lacks exact main prefix/target counterpart')
        source = seen[key]
        for field in ('losses', 'argmax'):
            _require(native._bits_equal(main[field][source['case_index'], source['scored_rows']],
                supplemental[field][case['case_index'], case['scored_rows'][:3]]),
                'main/supplemental first-three native scores differ')
        count += 1
    return count


def _positions(kind, length):
    result = {'selected_sequence': list(range(length))}
    result.update({f'token_{i}': [i] for i in range(length)})
    if kind in ('word', 'word_next_native'):
        result.update(first_piece=[0], suffix=[1, 2], word_three=[0, 1, 2])
        if kind == 'word_next_native':
            result['exact_next_native_token'] = [3]
    return result


def _aggregates(items, *, layout='complement'):
    cells = cube_subsets(layout=layout)
    grouped = defaultdict(list)
    for item in items:
        _require(set(item['cells']) == set(cells), 'aggregate case has incomplete or extra cube cells')
        for domain in dict.fromkeys((item['prefix_domain'], 'deduplicated_all')):
            key = (item['suite'], item['kind'], item['split'], item.get('spelling_variant'),
                   domain, item.get('word_has_leading_space'), item.get('piece_id'), item['target'])
            grouped[key].append(item)
    result = []
    for key, cases in sorted(grouped.items(), key=lambda pair: repr(pair[0])):
        metrics = {}
        for name, positions in _positions(cases[0]['kind'], len(cases[0]['target_ids'])).items():
            unique = {}
            for item in cases:
                identity = (item['prefix_sha256'], item['prefix_length'], tuple(item['target_ids'][:max(positions)+1]))
                values = {cell: math.fsum(item['cells'][cell]['token_log_probability'][p] for p in positions)
                          for cell in cells}
                _require(unique.setdefault(identity, values) == values, 'aliased metric has inconsistent native scores')
            means = {cell: math.fsum(v[cell] for v in unique.values())/len(unique) for cell in cells}
            metrics[name] = dict(unique_event_count=len(unique), alias_case_count=len(cases),
                cells={cell: dict(mean_log_probability=value, geometric_mean_probability=math.exp(value),
                    arithmetic_mean_probability=math.fsum(math.exp(v[cell]) for v in unique.values())/len(unique))
                    for cell, value in means.items()}, effects=score_cube(means, layout=layout))
        fields = ('suite', 'kind', 'split', 'spelling_variant', 'prefix_domain', 'word_has_leading_space', 'piece_id', 'target')
        result.append(dict(zip(fields, key), metrics=metrics))
    return result


def analyze_cube(models, cases_paths, scores, executions, output, *, rows,
                 config=GPT2Config(), layout='complement', case_archive=None):
    """CPU-only complete-cube readout; requires successful ledgers for all cells.

    A runner must separately pass E/EC all-row copy controls and FP64 reference
    comparisons before declaring a stage successful. This readout does not
    fabricate those gates or imply they were performed.
    """
    output = Path(output); cells = cube_subsets(layout=layout)
    selected_groups = groups(config, layout=layout)
    _require(not output.exists() and not output.is_symlink(), 'output already exists')
    _require(set(models) == set(scores) == set(executions) == set(cells)
             and set(cases_paths) == {'main', 'supplemental'}, 'incomplete cube input mapping')
    records, loaded = {}, {}
    endpoints = None
    for cell, subset in cells.items():
        model = models[cell]
        sources = (model['paths']['recipient'], model['paths']['donor'])
        endpoints = sources if endpoints is None else endpoints
        _require(sources == endpoints, 'cube changed source endpoints')
        _require(set(scores[cell]) == set(executions[cell]) == set(cases_paths), 'incomplete native suites')
        forbidden = [*model['paths'].values(), *scores[cell].values()]
        _require(all(Path(p).resolve() != output.resolve() and Path(p).resolve() not in output.resolve().parents
                     for p in forbidden), 'readout output overlaps checkpoints/native scores')
        checked = validate_model(model['patch']['path'], *sources, subset, rows,
                                 config=config, layout=layout)
        _require(checked == model, 'cube model no longer matches validated construction')
        loaded[cell] = {}
        for suite in cases_paths:
            archive_options = {} if case_archive is None else {'case_archive': case_archive}
            value = load_native(checked, cases_paths[suite], suite, scores[cell][suite],
                                executions[cell][suite], config=config, **archive_options)
            loaded[cell][suite] = value
            for record in value['records']:
                native._register(records, record['path'], record)
        _cross_suite(loaded[cell]['main'], loaded[cell]['supplemental'])
    per_case = []
    for suite in ('main', 'supplemental'):
        cases = loaded['E'][suite]['cases']
        amended = cases['plan']['format'] in branch.case_contract.AMENDED
        for case in cases['plan']['cases']:
            index = case['case_index']; selected = case['scored_rows']
            item = {k: case[k] for k in ('kind', 'split', 'context_id', 'prefix_domain', 'target', 'target_ids')}
            item.update(branch.case_contract.metadata(case, amended=amended))
            item.update(suite=suite, case_index=index, selected_rows=selected,
                prefix_sha256=case['prefix']['token_ids_sha256'], prefix_length=case['prefix']['length'], cells={})
            if case['kind'] in ('word', 'word_next_native'):
                item['word_has_leading_space'] = bytes.fromhex(case['target_source']['native_piece_bytes_hex'][0]).startswith(b' ')
            if 'piece_id' in case:
                item['piece_id'] = case['piece_id']
            for cell in cells:
                value = loaded[cell][suite]
                nll = [float(x) for x in value['losses'][index, selected]]
                winners = [int(x) for x in value['argmax'][index, selected]]
                item['cells'][cell] = dict(token_nll=nll, token_log_probability=[-x for x in nll],
                    token_probability=[math.exp(-x) for x in nll], argmax_ids=winners,
                    argmax_matches=[a == b for a, b in zip(winners, case['target_ids'])])
            item['metrics'] = {}
            for name, positions in _positions(case['kind'], len(selected)).items():
                values = {cell: math.fsum(item['cells'][cell]['token_log_probability'][p] for p in positions) for cell in cells}
                item['metrics'][name] = dict(cells={cell: dict(log_probability=v, probability=math.exp(v)) for cell, v in values.items()},
                                             effects=score_cube(values, layout=layout))
            per_case.append(item)
    for module in (__file__, native.__file__, branch.__file__, patcher.__file__):
        native._register(records, module)
    result = dict(format=FORMAT if layout == 'complement' else EARLY_BRANCHES_FORMAT,
        phase='cube_readout', complete=True, groups=selected_groups,
        cells={name: list(subset) for name, subset in cells.items()}, rows=sorted(rows),
        per_case=per_case, groups_readout=_aggregates(per_case, layout=layout),
        models={cell: dict(patch=m['patch'], paths=m['paths'], subset=m['subset']) for cell, m in models.items()},
        files=[records[p] for p in sorted(records)], model_forward_performed=False,
        baseline_controls_certified=False, goal_completion_claimed=False,
        definitions=dict(cells='E always fixes selected donor embedding rows in both roles; cell labels list extra donor groups.',
            effects='addition=s(E+G)-s(E); removal=s(EC)-s(EC-G); pair interactions condition on other group absent/present; three_way is their difference.',
            metrics='Natural-log teacher-forced probability; suffix conditions on supplied first ID; fourth is one exact following native token.',
            aliases='Each metric deduplicates actual prefix plus target IDs through its last target within its declared stratum; overlapping domains are not independent samples.'),
        limitations=['Coordinate-defined interventions, not unique storage or context-independent necessity.',
            'No GPU execution or baseline self-control claim is made by this CPU reader.',
            'Losses are native FP32 outputs, not full-vocabulary logit/margin decompositions.'])
    if layout == 'early_branches':
        result['layout'] = layout
        result['definitions']['groups'] = (
            'A = early attention + LN1; M = early MLP + LN2; '
            'R = positions + late blocks + final LN; EC = A+M+R. '
            'E remains fixed in both tied roles in every cell.')
        result['limitations'].extend([
            'Attention/MLP groups include their pre-LN affine parameters and projection biases.',
            'Early attention and MLP branches are interleaved residual computations, not parallel independent modules.',
            'R includes upstream position embeddings; it is not a purely downstream background.'])
    if case_archive is not None:
        result['case_source_archive'] = case_archive.manifest_record
    training.verify_records(result['files'])
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return result
