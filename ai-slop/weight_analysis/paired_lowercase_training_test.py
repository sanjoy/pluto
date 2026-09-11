"""CPU-only handoff contract tests; fixtures never launch a trainer or CUDA."""

from contextlib import ExitStack
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from . import checkpoint
from . import paired_lowercase_training as runner


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def identity(pid=10, ticks=20, argv=None, state='S'):
    return dict(pid=pid, start_ticks=ticks, argv=argv or ['known', '--arg'], state=state, parent_pid=1)


class ProcessTest(unittest.TestCase):
    def test_benign_running_sleeping_transition_is_not_identity_change(self):
        fields=['S','1']+['0']*17+['20']
        before='10 (python) '+' '.join(fields)
        fields[0]='R'
        after='10 (python) '+' '.join(fields)
        with mock.patch.object(Path,'read_text',side_effect=[before,after]), mock.patch.object(Path,'read_bytes',return_value=b'known\0--arg\0'):
            actual=runner.process_identity(10)
        self.assertEqual(actual,identity(state='R'))

    def test_same_process_and_zombie(self):
        expected = identity()
        with mock.patch.object(runner, 'process_identity', return_value=expected):
            self.assertTrue(runner.process_live(expected))
        with mock.patch.object(runner, 'process_identity', return_value=identity(state='Z', argv=[])):
            self.assertFalse(runner.process_live(expected))

    def test_changed_pid_or_command_rejected(self):
        for actual in (identity(ticks=21), identity(argv=['restarted'])):
            with self.subTest(actual=actual), mock.patch.object(runner, 'process_identity', return_value=actual):
                with self.assertRaises(RuntimeError):
                    runner.process_live(identity())

    def test_missing_proc_requires_independent_existence_check(self):
        with mock.patch.object(runner, 'process_identity', side_effect=FileNotFoundError):
            with mock.patch.object(runner.os, 'kill', side_effect=ProcessLookupError):
                self.assertFalse(runner.process_live(identity()))
            with mock.patch.object(runner.os, 'kill', return_value=None):
                with self.assertRaises(OSError):
                    runner.process_live(identity())

    def test_wait_both_and_two_consecutive_exits(self):
        with mock.patch.object(runner, 'process_live', side_effect=[True, False, False, False, False, False]) as live:
            with mock.patch.object(runner.time, 'sleep') as sleep:
                runner.wait_for_exit([identity(), identity(11)], poll_seconds=1)
        self.assertEqual(live.call_count, 6)
        self.assertEqual(sleep.call_count, 2)

    def test_transient_error_resets_exit_count(self):
        values = [False, False, OSError('read timeout'), False, False, False, False]
        with mock.patch.object(runner, 'process_live', side_effect=values) as live:
            with mock.patch.object(runner.time, 'sleep'):
                runner.wait_for_exit([identity(), identity(11)], poll_seconds=1)
        self.assertEqual(live.call_count, 7)

    def test_repeated_observation_failure_fails_closed(self):
        with mock.patch.object(runner, 'process_live', side_effect=OSError('timeout')):
            with mock.patch.object(runner.time, 'sleep'), mock.patch.object(runner.subprocess, 'Popen') as launch:
                with self.assertRaises(RuntimeError):
                    runner.wait_for_exit([identity()], poll_seconds=1)
                launch.assert_not_called()

    def test_invalid_wait_interval(self):
        for seconds in (0, -1, 61):
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                runner.wait_for_exit([identity()], poll_seconds=seconds)


class GpuGateTest(unittest.TestCase):
    def test_gpu_queries_and_visibility(self):
        answers = [mock.Mock(stdout='0, GPU-abc, GH200, 580\n'), mock.Mock(stdout='10, GPU-abc\n')]
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(runner.subprocess, 'run', side_effect=answers):
            result = runner.gpu_snapshot()
        self.assertEqual(result['processes'], [dict(pid=10, uuid='GPU-abc')])
        self.assertEqual(result['device']['driver_version'], '580')

    def test_disabled_cuda_rejected_before_launch(self):
        answer = mock.Mock(stdout='0, GPU-abc, GH200, 580\n')
        with mock.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}), mock.patch.object(runner.subprocess, 'run', return_value=answer):
            with self.assertRaises(ValueError):
                runner.gpu_snapshot()

    def test_busy_or_different_device_rejected(self):
        expected = dict(uuid='GPU-abc')
        for snapshot in (dict(device=expected, processes=[dict(pid=11)]),
                         dict(device=dict(uuid='GPU-other'), processes=[])):
            with self.subTest(snapshot=snapshot), mock.patch.object(runner, 'gpu_snapshot', return_value=snapshot):
                with self.assertRaises(RuntimeError):
                    runner.require_idle_gpu(expected)

    def test_idle_same_device_allowed(self):
        with mock.patch.object(runner, 'gpu_snapshot', return_value=dict(device={'uuid':'GPU-abc'}, processes=[])):
            runner.require_idle_gpu({'uuid':'GPU-abc'})


class FilesTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_record_rejects_links_and_detects_change(self):
        source = self.root / 'source'; source.write_bytes(b'old')
        rec = runner.record(source)
        link = self.root / 'link'; link.symlink_to(source)
        with self.assertRaises(ValueError): runner.record(link)
        source.write_bytes(b'new')
        with self.assertRaises(ValueError): runner.verify_records([rec])

    def checkpoint(self):
        config = checkpoint.GPT2Config(vocab_size=4, padded_vocab_size=4, context_length=4,
                                       n_layers=1, d_model=2, n_heads=1, d_ff=4)
        directory = self.root / 'checkpoints' / 'step_0'; directory.mkdir(parents=True)
        for spec in checkpoint.tensor_manifest(config):
            (directory / spec.filename).write_bytes(np.zeros(spec.shape, dtype='<f4').tobytes())
        return config, directory

    def test_inventory_real_finite_canonical_fixture(self):
        config, directory = self.checkpoint()
        result = runner.inventory(directory.parent, config=config)
        self.assertEqual([r['step'] for r in result], [0])
        self.assertEqual(len(result[0]['sha256']), 16)

    def test_inventory_rejects_extra_nonfinite_and_bad_names(self):
        config, directory = self.checkpoint()
        (directory / 'other.txt').write_text('extra')
        with self.assertRaises(ValueError): runner.inventory(directory.parent, config=config)
        (directory / 'other.txt').unlink()
        file = directory / 'weight_0.bin'
        values = np.fromfile(file, dtype='<f4'); values[0] = np.nan; file.write_bytes(values.tobytes())
        with self.assertRaises(ValueError): runner.inventory(directory.parent, config=config)

    def test_inventory_periodic_final_and_shared_init(self):
        hashes = {'weight_0.bin':'abc'}
        items = [dict(step=i, sha256=hashes) for i in (0, 100, 200, 233)]
        runner.validate_inventory(items, {'final_step':233}, hashes)
        for bad in (items[:-1], items[:1]+items[2:], [dict(step=0,sha256={})]+items[1:]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                runner.validate_inventory(bad, {'final_step':233}, hashes)
        with self.assertRaises(ValueError):
            runner.validate_inventory(items, {'final_step':233}, hashes, control=True)

    def test_control_requires_every_step(self):
        hashes = {'weight_0.bin':'abc'}
        runner.validate_inventory([dict(step=i, sha256=hashes) for i in (0,1,2)], {'final_step':2}, hashes, control=True)
        with self.assertRaises(ValueError):
            runner.validate_inventory([dict(step=i, sha256=hashes) for i in (0,2)], {'final_step':2}, hashes, control=True)


class HandoffTest(FilesTest):
    def setUp(self):
        super().setUp()
        self.output = self.root / 'lowercase_amendment'; self.output.mkdir()
        self.manifest = dict(root=str(self.root), flags={'seed':17,'batch_size':10}, seconds_per_arm=14400,
            inputs={'original.full':{'text':str(self.root/'original.txt')}}, binaries={'trainer':{'path':'/frozen/trainer'}})
        self.old_record = dict(path=str(self.root/'manifest.json'), bytes=1, sha256='original-manifest')
        self.guard = dict(format='pluto-replacement-handoff-guard-v1', original_training_must_continue=True,
            old_replacement_was_never_started=True, old_manifest_sha256='original-manifest',
            amendment_root=str(self.output), replacement_mappings={'Exeunt':'Nuveth','exeunt':'nuveth'})
        write(self.root/'replacement/SUPERSEDED_BEFORE_LAUNCH.json',self.guard)
        hashes={'weight_0.bin':'initial'}
        self.items=[dict(step=i,path=str(self.root/f'original/checkpoints/step_{i}'),sha256=hashes) for i in (0,100,200,250)]
        self.controls=[dict(step=i,path=f'/control/step_{i}',sha256=hashes) for i in (0,1,2)]
        self.gate=dict(status='verified',steps=[0,1,2],checkpoints={'control_a':self.controls,'control_b':self.controls})
        write(self.root/'determinism_gate.json',self.gate)
        self.request=dict(amendment_root=str(self.output),legacy_manifest=self.old_record,
            upstream_processes=[identity(10),identity(11)],frozen_inputs=[],initial_weights=hashes,determinism_gate=self.gate)
        self.terminal=dict(final_step=250,stop_reason='time_limit',training_elapsed_seconds=14401.0,training_loss=4.0,test_loss=4.1)
        argv=runner.paired_training.training_command(self.manifest,'original','original',seconds=14400)
        self.run=dict(command=argv,pid=11,returncode=0,elapsed_process_seconds=14420,
            started_utc='2026-09-10T11:00:00+00:00',finished_utc='2026-09-10T15:00:20+00:00',
            final_checkpoint=self.items[-1]['path'],initial_weights_match=True,**self.terminal)
        self.state=dict(phase='failed',runner_pid=10,error=f"FileExistsError: [Errno 17] File exists: '{self.root/'replacement'}'",
            runs={name:{} for name in ('initial','control_a','control_b','control_replacement')},determinism_gate=self.gate)
        self.state['runs']['original']=self.run
        write(self.root/'state.json',self.state)
        write(self.root/'original/command.json',argv)
        write(self.root/'original/checkpoints.json',self.items)
        (self.root/'original/process.log').write_text('process log')
        (self.root/'original/train.log').write_text('training stopped at step: 250\ntraining stop reason: time_limit\ntraining elapsed seconds: 14401\nfinal training loss: 4\nfinal test loss: 4.1\n')

    def validate(self):
        def inventory(path): return self.items if path.parent.name=='original' else self.controls
        with mock.patch.object(runner,'process_live',return_value=False), mock.patch.object(runner,'inventory',side_effect=inventory):
            return runner.validate_handoff(self.root,self.request,self.manifest)

    def test_exact_handoff_and_original_reference(self):
        result=self.validate()
        self.assertTrue(result['reused_not_rerun'])
        self.assertFalse(result['supervisor_exit_code_observed'])
        self.assertEqual(result['run']['command'],self.run['command'])
        self.assertEqual(len(result['evidence']),5)

    def test_live_upstream_cannot_handoff(self):
        with mock.patch.object(runner,'process_live',return_value=True), self.assertRaises(ValueError):
            runner.validate_handoff(self.root,self.request,self.manifest)

    def test_arbitrary_failure_or_launched_old_replacement_rejected(self):
        for mutation in ('error','replacement'):
            state=copy.deepcopy(self.state)
            if mutation=='error':state['error']='RuntimeError: crashed'
            else:state['runs']['replacement']={}
            write(self.root/'state.json',state)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): self.validate()

    def test_incomplete_time_budget_rejected(self):
        p=self.root/'original/train.log'
        p.write_text(p.read_text().replace('14401','14399'))
        with self.assertRaises(ValueError): self.validate()

    def test_checkpoint_inventory_does_not_match_published_evidence(self):
        write(self.root/'original/checkpoints.json',self.items[:-1])
        with self.assertRaises(ValueError): self.validate()

    def test_guard_must_be_only_legacy_replacement_content(self):
        (self.root/'replacement/command.json').write_text('[]')
        with self.assertRaises(ValueError): self.validate()

    def test_original_command_or_pid_differ_rejected(self):
        for key,value in (('pid',12),('command',['wrong'])):
            state=copy.deepcopy(self.state);state['runs']['original'][key]=value
            write(self.root/'state.json',state)
            with self.subTest(key=key), self.assertRaises(ValueError): self.validate()

    def test_child_command_uses_amended_corpus_and_old_initial(self):
        amendment={'inputs':{'replacement.full':{'text':'/amended/text'}}}
        argv=runner.command(self.manifest,amendment,self.output/'replacement')
        self.assertIn('--corpus=/amended/text',argv)
        self.assertIn('--resume_from='+str(self.root/'initial/checkpoints'),argv)
        self.assertIn('--training_seconds=14400',argv)
        self.assertFalse(any(s.startswith('--steps=') for s in argv))
        control=runner.command(self.manifest,amendment,self.output/'control_replacement',control=True)
        self.assertIn('--steps=2',control)
        self.assertIn('--checkpoint_every=1',control)

    def test_float_time_budget_preserves_original_argv_exactly(self):
        manifest=copy.deepcopy(self.manifest);manifest['seconds_per_arm']=14400.0
        original=runner.paired_training.training_command(manifest,'original','original',seconds=manifest['seconds_per_arm'])
        self.assertIn('--training_seconds=14400.0',original)
        amended=runner.command(manifest,{'inputs':{'replacement.full':{'text':'/amended'}}},self.output/'replacement')
        self.assertIn('--training_seconds=14400.0',amended)

    def test_child_launch_blocked_on_busy_gpu(self):
        state={'runs':{}}
        req={'frozen_inputs':[],'gpu':{'uuid':'GPU-abc'}}
        with mock.patch.object(runner,'require_idle_gpu',side_effect=RuntimeError('busy')):
            with mock.patch.object(runner.subprocess,'Popen') as launch:
                with self.assertRaises(RuntimeError):
                    runner.run_child(self.output,'replacement',['trainer'],state,req)
                launch.assert_not_called()
        self.assertFalse((self.output/'replacement').exists())

    def test_child_interruption_terminates_only_new_child(self):
        state={'runs':{}}
        request={'frozen_inputs':[],'gpu':{'uuid':'GPU-abc'}}
        child=mock.Mock(pid=1000)
        child.wait.side_effect=[KeyboardInterrupt('stop'),-15]
        with mock.patch.object(runner,'require_idle_gpu'), mock.patch.object(runner,'process_identity',return_value=identity(1000)):
            with mock.patch.object(runner.subprocess,'Popen',return_value=child), mock.patch.object(runner.os,'kill') as signal_old:
                with self.assertRaises(KeyboardInterrupt):
                    runner.run_child(self.output,'replacement',['trainer'],state,request)
                child.terminate.assert_called_once_with()
                signal_old.assert_not_called()
        self.assertEqual(state['runs']['replacement']['returncode'],-15)


class AmendmentValidationTest(FilesTest):
    def setUp(self):
        super().setUp()
        self.original_parts=[b' Ex',b'e',b'unt',b' ex',b'e',b'unt',b'\n',b'zzzzzzzzzz',b'\n']
        self.replaced_parts=[b' Nu',b've',b'th',b' nu',b've',b'th',b'\n',b'zzzzzzzzzz',b'\n']
        self.original_ids=[1475,68,2797,409,68,2797,198,1,198]
        self.replaced_ids=[21733,303,400,14364,303,400,198,1,198]
        self.boundary=sum(map(len,self.original_parts[:7]))
        self.manifest=dict(split_byte=self.boundary,flags={'test_fraction':.5},inputs={})
        self.old_record={'path':'/old/manifest','bytes':100,'sha256':'old'}
        self.amendment=dict(format='pluto-paired-lowercase-amendment-v1',complete=True,
            old_manifest=self.old_record,split_byte=self.boundary,inputs={},alignment={})
        for split,begin,end in (('full',0,9),('training',0,7),('test',7,9)):
            for domain,parts,ids in (('original',self.original_parts,self.original_ids),('replacement',self.replaced_parts,self.replaced_ids)):
                item=self.make_input(domain+'.'+split,parts[begin:end],ids[begin:end])
                self.amendment['inputs'][domain+'.'+split]=item
                if domain=='original':self.manifest['inputs'][domain+'.'+split]=item
            self.amendment['alignment'][split]={'token_count':end-begin,'replacements':0 if split=='test' else 2}

    def make_input(self,name,parts,ids):
        base=self.root/name
        text=Path(str(base)+'.txt');text.write_bytes(b''.join(parts))
        tokens=Path(str(base)+'.tokens.bin');tokens.write_bytes(np.asarray(ids,dtype='<u4').tobytes())
        offsets=Path(str(base)+'.offsets.bin')
        positions=[0]
        for part in parts:positions.append(positions[-1]+len(part))
        offsets.write_bytes(np.asarray(positions,dtype='<u8').tobytes())
        item={key:str(value) for key,value in (('text',text),('token_ids',tokens),('offsets',offsets))}
        item.update({key+'_sha256':runner.record(item[key])['sha256'] for key in ('text','token_ids','offsets')})
        item['export']=dict(token_dtype='<u4',offset_dtype='<u8',roundtrip_verified=True,
            token_count=len(ids),offset_count=len(positions),corpus_bytes=sum(map(len,parts)))
        return item

    def validate(self):
        return runner.validate_amendment(self.amendment,self.manifest,self.old_record)

    def rehash(self,item,key):
        item[key+'_sha256']=runner.record(item[key])['sha256']

    def test_exact_two_case_native_alignment(self):
        self.assertEqual(len(self.validate()),18)

    def test_changed_text_rejected_even_when_rehashed(self):
        item=self.amendment['inputs']['replacement.full']
        path=Path(item['text']);path.write_bytes(path.read_bytes().replace(b'nuveth',b'exeunt'))
        self.rehash(item,'text')
        with self.assertRaises(ValueError):self.validate()

    def test_changed_token_outside_replacement_rejected_even_when_rehashed(self):
        item=self.amendment['inputs']['replacement.full']
        ids=np.fromfile(item['token_ids'],dtype='<u4');ids[-2]=2
        Path(item['token_ids']).write_bytes(ids.tobytes());self.rehash(item,'token_ids')
        with self.assertRaises(ValueError):self.validate()

    def test_trailing_native_bytes_not_silently_ignored(self):
        item=self.amendment['inputs']['replacement.full']
        path=Path(item['token_ids']);path.write_bytes(path.read_bytes()+b'!');self.rehash(item,'token_ids')
        with self.assertRaises(ValueError):self.validate()

    def test_wrong_split_and_lineage_rejected(self):
        for key,value in (('split_byte',self.boundary-1),('old_manifest',{'sha256':'other'})):
            amended=copy.deepcopy(self.amendment);amended[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                runner.validate_amendment(amended,self.manifest,self.old_record)

    def test_changed_offsets_outside_replacement_rejected(self):
        item=self.amendment['inputs']['replacement.full']
        offsets=np.fromfile(item['offsets'],dtype='<u8');offsets[-2]-=1
        Path(item['offsets']).write_bytes(offsets.tobytes());self.rehash(item,'offsets')
        with self.assertRaises(ValueError):self.validate()


if __name__=='__main__':
    unittest.main()
