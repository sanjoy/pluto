"""CPU-only safety tests for explicit recovery of a loader-failed analysis."""

import copy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from . import paired_causal_recovery as recovery

training = recovery.training


class RecoveryGatesTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.previous = self.root / 'failed'
        (self.previous / 'plan').mkdir(parents=True)
        self.identity = dict(pid=100, start_ticks=200, argv=['old-runner'])
        self.request = dict(format=recovery.screen.FORMAT,
                            amendment_root=str(self.root),
                            stage='embedding_and_branches',
                            runner_identity=self.identity,
                            upstream_identity=dict(self.identity, pid=101),
                            frozen_inputs=[])
        self.state = dict(phase='gpu_validation', completed=[], runner_pid=100)
        self.failure = dict(type='RuntimeError', training_restarted=False,
                            error=f'analysis child exited 127; inspect {self.previous / "gpu_test.log"}')
        self.plan = dict(interventions=[dict(name='copy_original')])
        for name, value in (('request.json', self.request), ('state.json', self.state),
                            ('failure.json', self.failure), ('plan/plan.json', self.plan)):
            training.publish(self.previous / name, value)
        (self.previous / 'gpu_test.log').write_text(
            'error while loading shared libraries: missing.so')
        self.live = self.enterContext(mock.patch.object(training, 'process_live', return_value=False))

    def replace(self, name, value):
        training.publish(self.previous / name, value, exclusive=False)

    def validate(self):
        return recovery.validate_failure(self.root, self.previous)

    def test_loader_failure_requires_both_real_processes_exited(self):
        request, plan, records = self.validate()
        self.assertEqual(request, self.request)
        self.assertEqual(plan, self.plan)
        self.assertEqual(self.live.call_count, 2)
        training.verify_records(records)

    def test_live_transient_or_reused_pid_never_authorizes_relaunch(self):
        for error in (OSError('transient read'), RuntimeError('PID reused')):
            self.live.side_effect = error
            with self.assertRaises(type(error)):
                self.validate()
        self.live.side_effect = None
        self.live.return_value = True
        with self.assertRaisesRegex(RuntimeError, 'still live'):
            self.validate()

    def test_intervention_or_completed_execution_rejected(self):
        (self.previous / 'copy_original').mkdir()
        with self.assertRaisesRegex(ValueError, 'materialized'):
            self.validate()
        (self.previous / 'copy_original').rmdir()
        training.publish(self.previous / 'gpu_test_execution.json', {})
        with self.assertRaisesRegex(ValueError, 'completed execution'):
            self.validate()

    def test_wrong_failure_or_state_rejected(self):
        self.replace('failure.json', dict(self.failure, error='other error'))
        with self.assertRaisesRegex(ValueError, 'loader-only'):
            self.validate()
        self.replace('failure.json', self.failure)
        for field, value in (('phase', 'interventions'), ('completed', ['copy_original']),
                             ('runner_pid', 999)):
            self.replace('state.json', dict(self.state, **{field: value}))
            with self.assertRaisesRegex(ValueError, 'loader-only'):
                self.validate()

    def test_mutated_frozen_evidence_rejected(self):
        source = self.root / 'frozen'
        source.write_text('before')
        self.replace('request.json', dict(self.request, frozen_inputs=[training.record(source)]))
        source.write_text('after')
        with self.assertRaisesRegex(ValueError, 'frozen evidence'):
            self.validate()

    def test_fresh_plan_must_match_every_field(self):
        recovery.require_same_plan(copy.deepcopy(self.plan), self.plan)
        for actual in (dict(self.plan, extra=True), dict(interventions=[])):
            with self.assertRaisesRegex(ValueError, 'predeclared plan'):
                recovery.require_same_plan(actual, self.plan)

    def test_no_overwrite_of_existing_output(self):
        args = SimpleNamespace(root=self.root, previous=self.previous, output=self.previous)
        with mock.patch.object(recovery, 'validate_failure') as validate:
            with self.assertRaisesRegex(ValueError, 'NEW'):
                recovery.run(args)
            validate.assert_not_called()

    def test_downstream_failure_contract_and_actual_exit(self):
        name = self.root / 'historical_head_position_stage'
        name.mkdir()
        training.publish(name / 'request.json', dict(runner_identity=self.identity))
        training.publish(name / 'failure.json', dict(training_restarted=False,
            error='historical source stage failed; no GPU work authorized'))
        records = recovery.check_downstream(self.root)
        self.assertEqual(len(records), 2)
        training.verify_records(records)
        self.live.return_value = True
        with self.assertRaisesRegex(RuntimeError, 'still live'):
            recovery.check_downstream(self.root)


if __name__ == '__main__':
    unittest.main()
