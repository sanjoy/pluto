"""Synthetic arithmetic/integrity tests; no real checkpoints, corpus, or model."""

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from . import delta_ordering as order
from . import restart_delta
from .checkpoint import GPT2Config


def unit_rows(values):
    result=np.asarray(values,dtype=np.float64).copy()
    for row in result:
        length=sum(float(x*x) for x in row)**.5
        if length:
            row/=length
    return result


def explicit_association(centered,residual,permutation,arm):
    """Scalar all-pairs oracle, written in source/target rather than BLAS form."""
    vocabulary=len(centered[0])
    result=np.empty((vocabulary,vocabulary))
    for source in range(vocabulary):
        for target in range(vocabulary):
            values=[]
            for c,r in zip(centered,residual):
                if arm=='directional': left,right=c[source],r[target]
                elif arm=='transpose': left,right=c[target],r[source]
                elif arm=='static': left,right=c[source],c[target]
                else: left,right=c[source],r[permutation[target]]
                values.append(sum(float(x*y) for x,y in zip(left,right)))
            result[source,target]=sum(values)/len(values)
    return result


class GeometryTest(unittest.TestCase):
    def test_matches_existing_adjustment_and_independent_scalar_fit(self):
        before=np.array([[1.,2],[3,0],[-2,4],[0,-1]],dtype=np.float32)
        after=np.array([[2.,4],[2,1],[-3,7],[1,2]],dtype=np.float32)
        snapshots=(before.tobytes(),after.tobytes())
        c,r,cn,rn,summary=order.normalized_geometry(before,after)
        previous,prior_summary=restart_delta.activities(before,after)
        np.testing.assert_array_equal(rn,previous['adjusted'])
        for name in ('shared_translation','fitted_radial_coefficient_not_authenticated_decay',
                     'centered_before_squared_norm'):
            self.assertEqual(summary[name],prior_summary[name])
        c0=before.astype(float)-sum(before.astype(float))/4
        d=after.astype(float)-before
        translation=sum(d)/4
        coefficient=sum(float(x@y) for x,y in zip(c0,d-translation))/sum(float(x@x) for x in c0)
        r0=d-translation-coefficient*c0
        np.testing.assert_allclose(c,unit_rows(c0),atol=2e-15)
        np.testing.assert_allclose(r,unit_rows(r0),atol=2e-15)
        np.testing.assert_allclose(cn,[sum(float(x*x) for x in row)**.5 for row in c0])
        self.assertEqual((before.tobytes(),after.tobytes()),snapshots)

    def test_zero_constant_and_tiny_rows_no_silent_floor(self):
        c,r,cn,rn,summary=order.normalized_geometry(np.ones((4,2)),np.ones((4,2))*3)
        np.testing.assert_array_equal(c,np.zeros((4,2)))
        np.testing.assert_array_equal(r,np.zeros((4,2)))
        self.assertEqual(summary['centered_zero_rows'],4)
        self.assertEqual(summary['residual_zero_rows'],4)
        before=np.array([[1.,0],[0,1],[-1,0],[0,-1]])
        after=before+1e-13*np.array([[0,1],[-1,0],[0,-1],[1,0]])
        _,r,_,rn,_=order.normalized_geometry(before,after)
        self.assertTrue(np.all((rn>0)&(rn<1e-12)))
        np.testing.assert_allclose(np.linalg.norm(r,axis=1),1)

    def test_pure_rotation_is_directed_without_token_order_information(self):
        before=np.array([[1.,0],[0,1],[-1,0],[0,-1]])
        after=before@np.array([[0.,1],[-1,0]])
        c,r,_,_,_=order.normalized_geometry(before,after)
        association=c@r.T
        np.testing.assert_array_equal(association,-association.T)
        self.assertEqual(np.max(np.abs(association)),1)

    def test_rejects_invalid_shapes_and_nonfinite_values(self):
        for a,b in (([],[]),([[1]],[[1,2]]),([[np.nan]],[[0]]),([[0]],[[np.inf]])):
            with self.assertRaises(ValueError):order.normalized_geometry(a,b)


class AssociationTest(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(19)
        self.c=[unit_rows(rng.normal(size=(9,3))) for _ in range(2)]
        self.r=[unit_rows(rng.normal(size=(9,3))) for _ in range(2)]
        self.permutation=np.array([6,2,7,1,5,0,8,4,3])
        self.oracle=order.Associations(self.c,self.r,self.permutation)

    def test_all_arms_both_directions_equal_independent_dense_matrices(self):
        fixed=[2,0,7]
        for arm in order.ARMS:
            full=explicit_association(self.c,self.r,self.permutation,arm)
            for direction in ('predecessor','successor'):
                scores,components=self.oracle.scores(arm,direction,fixed)
                expected=full[:,fixed] if direction=='predecessor' else full[fixed,:].T
                np.testing.assert_allclose(scores,expected,atol=2e-15)
                self.assertEqual(components.shape,(2,9,3))
                for index in range(2):
                    individual=explicit_association([self.c[index]],[self.r[index]],self.permutation,arm)
                    expected=individual[:,fixed] if direction=='predecessor' else individual[fixed,:].T
                    np.testing.assert_allclose(components[index],expected,atol=2e-15)

    def test_all_paths_match_scalar_search_orientation_and_keep_all_seeds(self):
        seeds=[7,3,0]
        for arm in order.ARMS:
            full=explicit_association(self.c,self.r,self.permutation,arm)
            candidates=order.greedy_paths(self.oracle,arm,seeds)
            self.assertEqual(len(candidates),len(seeds))
            for seed,candidate in zip(seeds,candidates):
                expected=[seed]
                for direction in ['predecessor']*3+['successor']*4:
                    eligible=[token for token in range(9) if token not in expected]
                    if direction=='predecessor':
                        best=min(eligible,key=lambda token:(-full[token,expected[0]],token))
                        expected.insert(0,best)
                    else:
                        best=min(eligible,key=lambda token:(-full[expected[-1],token],token))
                        expected.append(best)
                self.assertEqual(candidate['token_ids'],expected)
                self.assertEqual(expected[3],seed)
                self.assertEqual(len(set(expected)),8)
                self.assertEqual([x['position'] for x in candidate['edges']],list(range(7)))
                for index,edge in enumerate(candidate['edges']):
                    self.assertEqual([edge['source_id'],edge['target_id']],expected[index:index+2])
                    self.assertAlmostEqual(edge['score'],full[expected[index],expected[index+1]])
                    self.assertAlmostEqual(edge['score'],sum(edge['interval_components'])/2)
                self.assertAlmostEqual(candidate['path_score'],sum(full[s,t] for s,t in zip(expected,expected[1:])))

    def test_exact_ties_ascending_ids_zero_rows_and_per_path_exclusions(self):
        zeros=[np.zeros((9,2))]*2
        oracle=order.Associations(zeros,zeros,np.arange(9))
        for arm in order.ARMS:
            result=order.greedy_paths(oracle,arm,[7,8])
            self.assertEqual(result[0]['token_ids'],[2,1,0,7,3,4,5,6])
            self.assertEqual(result[1]['token_ids'],[2,1,0,8,3,4,5,6])
            self.assertTrue(all(edge['score']==0 for row in result for edge in row['edges']))
            self.assertTrue(all(row['seed_position']==3 for row in result))

    def test_invalid_association_permutation_queries_and_path_inputs(self):
        for permutation in ([0]*9,list(range(8)),np.arange(9,dtype=float)):
            with self.assertRaises(ValueError):order.Associations(self.c,self.r,permutation)
        with self.assertRaises(ValueError):order.Associations(self.c,self.r[:1],self.permutation)
        for arm,direction,fixed in [('bad','successor',[0]),('static','bad',[0]),
                                     ('static','successor',[]),('static','successor',[-1]),
                                     ('static','successor',[9]),('static','successor',[True])]:
            with self.assertRaises(ValueError):self.oracle.scores(arm,direction,fixed)
        for seeds,pre,suc in [([],3,4),([1,1],3,4),([True],3,4),([9],3,4),
                             ([1],-1,4),([1],True,4),([1],3,9)]:
            with self.assertRaises(ValueError):order.greedy_paths(self.oracle,'static',seeds,pre,suc)
        with self.assertRaises(ValueError):order.greedy_paths(self.oracle,'bad',[0],0,0)

    def test_physical_address_orientation_for_each_control(self):
        expected={'directional':(2,7),'transpose':(7,2),'static':(2,7),
                  'permuted_delta_labels':(2,int(self.permutation[7]))}
        for arm,(left,right) in expected.items():
            locations=order.address_sources(arm,2,7,self.permutation,3)
            self.assertEqual(locations['left_centered_before']['byte_interval'],[12*left,12*(left+1)])
            self.assertEqual(locations['right_before']['byte_interval'],[12*right,12*(right+1)])
            self.assertEqual(locations['right_after'] is None,arm=='static')


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='delta-ordering-test-')
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.cp=self.root/'checkpoints';self.cp.mkdir()
        self.output=self.root/'output'
        self.protocol=self.root/'protocol.md';self.protocol.write_text('Synthetic protocol.\n')
        self.seed_path=self.root/'seeds.json'
        self.seed_payload=dict(stage='frozen_unordered_token_candidates_from_weight_deltas',
                               top_k=128,candidate_lists={'shared_boundary_adjusted':{'token_ids':list(range(128))}})
        self.seed_path.write_text(json.dumps(self.seed_payload))
        self.config=GPT2Config(vocab_size=132,padded_vocab_size=136,d_model=4,
                               n_heads=1,d_ff=8,n_layers=1,context_length=4)
        self.steps=[step for b in order.BOUNDARIES for step in (b,b+10)]
        for step in self.steps:(self.cp/f'step_{step}.tar.gz').write_bytes(f'synthetic {step}'.encode())
        self.reads=[]

    def reader(self,path,step):
        self.assertTrue((self.output/'plan.json').is_file())
        self.assertFalse((self.output/'frozen.json').exists())
        self.reads.append(step)
        values=np.random.default_rng(step).normal(size=(136,4)).astype('<f4')
        values[132:]=1e12
        return values,dict(input_files=[order.file_record(path)],input_files_unchanged_during_read=True,
                           embedding_sha256=hashlib.sha256(values.tobytes()).hexdigest())

    def pipeline(self,reader=None):
        with mock.patch.object(order,'GPT2Config',return_value=self.config), \
             mock.patch.object(order,'SEED_CANDIDATES_SHA256',order.file_record(self.seed_path)['sha256']), \
             mock.patch.object(order,'read_embedding',side_effect=reader or self.reader), \
             mock.patch.dict(os.environ,OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4'), \
             contextlib.redirect_stdout(io.StringIO()):
            return order.run(self.cp,self.output,self.protocol,self.seed_path)

    def test_full_freeze_candidate_schema_addresses_norms_hashes_and_padding(self):
        frozen=self.pipeline()
        self.assertEqual(self.reads,self.steps)
        self.assertEqual(frozen['candidate_count'],512)
        self.assertEqual(len(frozen['checkpoint_provenance']),10)
        self.assertEqual(read_json(self.output/'frozen.json'),frozen)
        rows=[json.loads(line) for line in (self.output/'candidates.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),512)
        permutation=np.load(self.output/'permutation.npy',allow_pickle=False)
        np.testing.assert_array_equal(permutation,np.random.Generator(np.random.PCG64(20260909)).permutation(132))
        for arm in order.ARMS:
            subset=[row for row in rows if row['method']==arm]
            self.assertEqual([row['seed_id'] for row in subset],list(range(128)))
            for row in subset:
                self.assertEqual(row['token_ids'][3],row['seed_id'])
                self.assertEqual(len(set(row['token_ids'])),8)
                self.assertTrue(all(0<=token<132 for token in row['token_ids']))
                for edge in row['edges']:
                    addresses=order.address_sources(arm,edge['source_id'],edge['target_id'],permutation,4)
                    self.assertEqual(edge['physical_weight_row_sources'],addresses)
                    self.assertEqual(len(edge['interval_components']),5)
                    self.assertAlmostEqual(edge['score'],sum(edge['interval_components'])/5)
                    for b,norms in zip(order.BOUNDARIES,edge['original_component_row_norms']):
                        with np.load(self.output/f'geometry_{b}_{b+10}.npz',allow_pickle=False) as saved:
                            cn=saved['centered_before_row_norms'];rn=saved['adjusted_delta_row_norms']
                            self.assertEqual(norms['left_centered_before'],cn[addresses['left_centered_before']['row_id']])
                            right=addresses['right_before']['row_id']
                            self.assertEqual(norms['right'],cn[right] if arm=='static' else rn[right])
        for record in [*frozen['sources'].values(),*frozen['files'].values()]:
            self.assertEqual(order.file_record(record['path']),record)
        with self.assertRaises(FileExistsError):self.pipeline()

    def test_plan_precedes_reads_and_reader_failure_cannot_freeze(self):
        def broken(path,step):
            self.reader(path,step)
            raise ValueError('synthetic failure')
        with self.assertRaisesRegex(ValueError,'synthetic failure'):self.pipeline(broken)
        self.assertTrue((self.output/'plan.json').exists())
        self.assertFalse((self.output/'frozen.json').exists())

    def test_invalid_paths_fail_before_any_weight_read(self):
        self.output=self.cp/'forbidden'
        with self.assertRaisesRegex(ValueError,'outside checkpoint'):self.pipeline()
        self.output=self.root/'output'
        (self.cp/f'step_{self.steps[0]}').mkdir()
        with self.assertRaisesRegex(ValueError,'missing or ambiguous'):self.pipeline()
        self.assertEqual(self.reads,[])

    def test_changed_checkpoint_protocol_and_output_prevent_freeze(self):
        for kind in ('checkpoint','protocol','output'):
            self.output=self.root/kind
            def changed(path,step):
                answer=self.reader(path,step)
                if step==self.steps[-1]:
                    target=(self.cp/f'step_{self.steps[0]}.tar.gz' if kind=='checkpoint'
                            else self.protocol if kind=='protocol' else self.output/'permutation.npy')
                    target.write_bytes(b'changed synthetic evidence')
                return answer
            with self.assertRaisesRegex(ValueError,'changed before freeze'):self.pipeline(changed)
            self.assertFalse((self.output/'frozen.json').exists())

    def test_unverified_or_wrong_shaped_reader_rejected(self):
        for kind in ('unverified','wrong_shape'):
            self.output=self.root/kind
            def changed(path,step):
                values,record=self.reader(path,step)
                if kind=='unverified':record['input_files_unchanged_during_read']=False
                else:values=values[:-1]
                return values,record
            with self.assertRaisesRegex(ValueError,'unverified or incorrectly shaped'):self.pipeline(changed)
            self.assertFalse((self.output/'frozen.json').exists())

    def test_seed_full_hash_shape_duplicates_and_json_guards(self):
        with self.assertRaisesRegex(ValueError,'hash does not match'):order.load_seeds(self.seed_path,132)
        for invalid in (list(range(127)),[0]*128,[True]+list(range(1,128)),list(range(127))+[132]):
            self.seed_payload['candidate_lists']['shared_boundary_adjusted']['token_ids']=invalid
            self.seed_path.write_text(json.dumps(self.seed_payload))
            with mock.patch.object(order,'SEED_CANDIDATES_SHA256',order.file_record(self.seed_path)['sha256']):
                with self.assertRaisesRegex(ValueError,'128 distinct'):order.load_seeds(self.seed_path,132)
        for text in ('{"x":1,"x":2}','{"x":NaN}','{"x":Infinity}','{"x":1e1000}'):
            with self.assertRaises(ValueError):order.strict_json(text)

    def test_cpu_thread_contract_checked_before_payload_reads(self):
        with mock.patch.dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4'):
            with self.assertRaisesRegex(ValueError,'OPENBLAS_NUM_THREADS=4'):
                order.run(self.cp,self.output,self.protocol,self.seed_path)
        self.assertEqual(self.reads,[])


def read_json(path):
    return json.loads(Path(path).read_text())


if __name__=='__main__':unittest.main()
