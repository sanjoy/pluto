"""Authenticate the historical suffix stage before head-position GPU work.

This is a CPU-only handoff helper, not a supervisor. It cannot launch native
programs or signal either existing controller. A summary does not substitute
for the exact publisher's confirmed exit. Once exited, the historical suffix
assay, native outputs and independent readout are revalidated, as is the entire
upstream paired-training/causal chain through its existing strong validator.
The old suffix measurement remains historical and NOT an initial-piece assay.
"""

from datetime import datetime, timezone
from pathlib import Path
import time

from . import paired_lowercase_source_value_followup as source
from . import paired_lowercase_training as training
from . import source_value_followup as historical
from . import source_value_readout as readout

MODULE = 'scripts.weight_analysis.paired_lowercase_source_value_followup'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def flag(argv, name):
    require(argv.count(name) == 1, 'missing/duplicate source observer flag: ' + name)
    index = argv.index(name)
    require(index + 1 < len(argv), 'missing source observer flag value: ' + name)
    return argv[index + 1]


def _source_kind(value):
    require(value.get('format') == source.FORMAT
            and value.get('stage') == 'historical_source_value'
            and value.get('new_paired_model_result') is False
            and value.get('initial_piece_assay') is False
            and value.get('goal_completion_claimed') is False,
            'requires historical suffix source-value stage, not a paired/initial-piece result')


def _request_identity(root, upstream, request):
    _source_kind(request)
    require(request.get('amendment_root') == str(root), 'source observer amendment root differs')
    saved = request['runner_identity']
    argv = saved['argv']
    require(type(saved['pid']) is int and saved['pid'] > 0
            and type(saved['start_ticks']) is int and saved['start_ticks'] >= 0
            and isinstance(argv, list) and all(type(a) is str for a in argv)
            and argv.count('-m') == 1 and argv.index('-m') + 1 < len(argv)
            and argv[argv.index('-m') + 1] == MODULE,
            'source request lacks the exact module/process identity')
    require(Path(flag(argv, '--root')).resolve() == root
            and Path(flag(argv, '--output')).resolve() == upstream,
            'source observer root/output differs')
    causal_output = Path(request['causal_output'])
    require(causal_output.parent == root and causal_output.resolve() == causal_output
            and Path(flag(argv, '--causal-output')).resolve() == causal_output,
            'source observer causal output differs')
    return saved


def observer_identity(root, upstream, pid, ticks, request):
    """Require the exact LIVE historical-source publisher named by its request."""
    root, upstream = Path(root).resolve(strict=True), Path(upstream).resolve(strict=True)
    require(upstream.parent == root, 'source output must be an immediate amended-root child')
    saved = _request_identity(root, upstream, request)
    actual = training.process_identity(pid)
    require(type(ticks) is int and actual['start_ticks'] == ticks
            and actual['state'] not in ('Z', 'X', 'x')
            and all(actual[k] == saved[k] for k in ('pid', 'start_ticks', 'argv')),
            'upstream is not the exact live historical source observer')
    return actual


def wait_for_source(summary_path, expected, *, poll_seconds=30):
    """Fail closed on identity changes, failed runs, or unreadable live handles."""
    require(0 < poll_seconds <= 60, 'poll interval must be in (0,60]')
    summary_path = Path(summary_path)
    missing = errors = 0
    while True:
        if (summary_path.parent / 'failure.json').exists():
            raise RuntimeError('historical source stage failed; no GPU work authorized')
        try:
            live = training.process_live(expected)
            errors = 0
        except (OSError, ValueError) as error:
            missing = 0
            errors += 1
            if errors >= 4:
                raise RuntimeError('cannot verify source observer handle') from error
            time.sleep(poll_seconds)
            continue
        missing = 0 if live else missing + 1
        if missing >= 2 and summary_path.is_file():
            summary = training.read_json(summary_path)
            _source_kind(summary)
            require(summary.get('complete') is True, 'source summary is incomplete')
            return summary
        if missing >= 4:
            raise RuntimeError('source observer exited without completed evidence; no restart')
        time.sleep(poll_seconds)


def _records(values, label, *, identical_duplicates=False):
    require(isinstance(values, list), label + ' is not a record list')
    result = {}
    for item in values:
        require(isinstance(item, dict) and set(item) == {'path', 'bytes', 'sha256'},
                label + ' contains a malformed record')
        path = item['path']
        require(path not in result or identical_duplicates and result[path] == item,
                label + ' contains duplicate/conflicting paths')
        result[path] = item
    training.verify_records(list(result.values()))
    return result


def _execution(path, expected_format, command, frozen, outputs, log):
    """Validate a completed child record; never execute its recorded command."""
    record = training.record(path)
    value = training.read_json(path)
    require(value.get('format') == expected_format and type(value.get('pid')) is int
            and value['pid'] > 0 and type(value.get('returncode')) is int
            and value['returncode'] == 0 and value.get('command') == command,
            'native execution identity/command differs')
    before = _records(value.get('inputs_before'), 'execution inputs')
    require(before == frozen and value.get('inputs_before') == value.get('inputs_after'),
            'native execution input ledger differs from frozen request')
    require(value.get('outputs') == outputs and value.get('log') == training.record(log),
            'native output/log ledger differs from actual files')
    _records(value['outputs'], 'execution outputs')
    times = [datetime.fromisoformat(value[key]) for key in ('started_utc', 'finished_utc')]
    require(all(t.tzinfo is not None and t.utcoffset() == timezone.utc.utcoffset(t) for t in times)
            and times[0] <= times[1], 'invalid native execution timestamps')
    return value, record, times


def validate_upstream(root, upstream, summary, output):
    """Return reverified records; create only NEW output/plan.json through reuse.

    The regenerated full-chain validator writes the same immutable causal plan
    to a new location. Only THAT record's path is normalized when comparing
    with the saved source-stage handoff ledger; every old record remains bound
    to its actual bytes. No successful outside-child exit code is invented.
    This function assumes wait_for_source established publisher exit.
    """
    root, upstream = Path(root).resolve(strict=True), Path(upstream).resolve(strict=True)
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    require(upstream.parent == root and output.parent.resolve(strict=True) == output.parent
            and upstream not in output.parents and output != root,
            'new revalidation output overlaps source evidence or has a linked parent')
    require(not (upstream / 'failure.json').exists(), 'source stage has a failure marker')
    request_path, summary_path = upstream / 'request.json', upstream / 'summary.json'
    request_record, summary_record = training.record(request_path), training.record(summary_path)
    require(training.read_json(summary_path) == summary, 'source summary changed after handoff')
    request = training.read_json(request_path)
    saved = _request_identity(root, upstream, request)
    _source_kind(summary)
    require(summary.get('complete') is True and summary.get('frozen_inputs') == request['frozen_inputs']
            and summary.get('assay') == request['assay'], 'source summary/request disagree')
    state_path = upstream / 'state.json'
    state_record = training.record(state_path)
    state = training.read_json(state_path)
    require(state.get('format') == source.FORMAT and state.get('phase') == 'complete',
            'source publisher did not reach complete state')
    frozen = _records(request['frozen_inputs'], 'frozen source inputs', identical_duplicates=True)
    training_request = training.read_json(root / 'request.json')
    require(training.record(root / 'request.json') in request['frozen_inputs']
            and training_request['gpu'] == request['gpu'], 'source request lost training/GPU provenance')
    argv = saved['argv']
    assay, historical_records = historical.authenticate_assay(
        Path(flag(argv, '--historical-manifest')).resolve(strict=True),
        Path(flag(argv, '--historical-root')).resolve(strict=True),
        Path(flag(argv, '--historical-checkpoint')).resolve(strict=True))
    require(assay == request['assay'] and all(frozen.get(r['path']) == r for r in historical_records),
            'source assay is not the authenticated fixed historical suffix event')
    require(set(request['binaries']) == {'probe', 'gpu_test'}, 'source binary inventory differs')
    for name, argument in (('probe', '--probe'), ('gpu_test', '--gpu-test')):
        copy = training.record(upstream / 'bin' / name)
        original = training.record(Path(flag(argv, argument)).resolve(strict=True))
        require(copy == request['binaries'][name] and frozen.get(copy['path']) == copy
                and frozen.get(original['path']) == original
                and (copy['bytes'], copy['sha256']) == (original['bytes'], original['sha256']),
                'source binary original/copy disagrees with frozen request')

    test_file = upstream / 'gpu_validation' / 'gtest.json'
    test_outputs = historical.recursive_outputs(test_file.parent)
    test, test_record, test_times = _execution(upstream / 'gpu_test_execution.json',
        'pluto-paired-probe-execution-v1',
        [request['binaries']['gpu_test']['path'], f'--gtest_output=json:{test_file}'],
        frozen, test_outputs, upstream / 'gpu_test.log')
    require(summary.get('gpu_test') == test, 'source test record differs from summary')
    historical.causal.require_gpu_test_success(test_file)
    native = upstream / 'native'
    native_outputs = historical.recursive_outputs(native)
    command = [request['binaries']['probe']['path'],
        f'--checkpoint={assay["historical_checkpoint"]}', f'--tokens_file={assay["prefix"]["path"]}',
        f'--output_dir={native}', f'--block={assay["block"]}', f'--head={assay["head"]}',
        f'--query={assay["query"]}', f'--sources={",".join(map(str, assay["sources"]))}',
        f'--target_id={assay["target_id"]}', f'--expected_logits={assay["expected_logits"]["path"]}']
    execution, execution_record, execution_times = _execution(upstream / 'probe_execution.json',
        'pluto-source-value-execution-v1', command, frozen, native_outputs, upstream / 'probe.log')
    require(summary.get('execution') == execution_record and test_times[1] <= execution_times[0],
            'source execution record/order differs')
    readout_path = upstream / 'readout.json'
    readout_record = training.record(readout_path)
    require(summary.get('readout') == readout_record, 'source readout hash differs')
    regenerated_readout = readout.summarize(native, execution_record['path'], assay)
    require(training.read_json(readout_path) == regenerated_readout,
            'source readout differs from independently regenerated native result')

    ledger_path = upstream / 'validated_upstream.json'
    ledger_record = training.record(ledger_path)
    require(summary.get('upstream') == ledger_record, 'source upstream-ledger hash differs')
    ledger = training.read_json(ledger_path)
    require(ledger.get('observer_identity') == request['upstream_identity']
            and 'observer_exit_code' in ledger and ledger['observer_exit_code'] is None,
            'source upstream handoff identity differs or invents outside-child exit status')
    old_ledger = _records(ledger.get('inputs'), 'saved upstream ledger')
    causal_output = Path(request['causal_output'])
    causal_request = training.read_json(causal_output / 'request.json')
    require(training.record(causal_output / 'request.json') in request['frozen_inputs']
            and all(request['upstream_identity'][k] == causal_request['runner_identity'][k]
                    for k in ('pid', 'start_ticks', 'argv')),
            'source handoff does not name the actual combined publisher')
    causal_summary = training.read_json(causal_output / 'summary.json')
    # This re-runs completed-training, deterministic control, native-case,
    # checkpoint and every prescribed transfer gate. Do not replace it with
    # a superficial check of a couple of successful summary flags.
    chain_records = source.validate_upstream(root, causal_output, causal_summary, output)
    fresh_plan = training.record(output / 'plan.json')
    old_plan = training.record(upstream / 'upstream_revalidation_plan' / 'plan.json')
    require((fresh_plan['bytes'], fresh_plan['sha256']) == (old_plan['bytes'], old_plan['sha256']),
            'saved and freshly regenerated causal plans differ')
    normalized = [old_plan if r['path'] == fresh_plan['path'] else r for r in chain_records]
    require(sum(r['path'] == fresh_plan['path'] for r in chain_records) == 1
            and normalized == ledger['inputs'],
            'saved upstream ledger differs from actual revalidated causal-chain evidence')
    records = [request_record, summary_record, state_record, test_record,
               execution_record, readout_record, ledger_record, *frozen.values(),
               *historical_records, *test_outputs, *native_outputs, test['log'], execution['log'],
               *old_ledger.values(), *chain_records]
    unique = _records(records, 'complete handoff evidence', identical_duplicates=True)
    return list(unique.values())
