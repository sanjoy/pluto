"""Exact spelling-pair semantics independent of GPU and corpus geometry."""

import copy
import unittest
from unittest import mock

from . import paired_case_contract as contract


class CaseContractTest(unittest.TestCase):
    def test_legacy_missing_role_fields_normalize_to_title_only(self):
        for role,word in contract.PAIRS['title'].items():
            result=contract.metadata(dict(kind='word',target=word))
            self.assertEqual(result,dict(spelling_variant='title',candidate_pair=contract.PAIRS['title'],target_source_domain=role))
        with self.assertRaises(ValueError): contract.metadata(dict(kind='word',target='exeunt'))

    def test_amended_requires_all_three_explicit_identity_fields(self):
        for variant,pair in contract.PAIRS.items():
            for role,word in pair.items():
                case=dict(kind='word',target=word,spelling_variant=variant,candidate_pair=pair,target_source_domain=role)
                self.assertEqual(contract.metadata(case,amended=True)['target_source_domain'],role)
                for missing in ('spelling_variant','candidate_pair','target_source_domain'):
                    bad=dict(case); del bad[missing]
                    with self.subTest(variant=variant,role=role,missing=missing),self.assertRaises(ValueError):
                        contract.metadata(bad,amended=True)

    def test_lowercase_cannot_be_silently_accepted_as_legacy(self):
        case=dict(kind='word',target='exeunt',spelling_variant='lowercase',candidate_pair=contract.PAIRS['lowercase'],target_source_domain='original')
        with self.assertRaises(ValueError): contract.metadata(case)

    def test_control_cannot_claim_a_word_pair(self):
        self.assertIsNone(contract.metadata(dict(kind='control',target='control_next_3'))['candidate_pair'])
        for extra in (dict(candidate_pair=contract.PAIRS['title']),dict(spelling_variant='lowercase')):
            with self.assertRaises(ValueError): contract.metadata(dict(kind='shared_piece',**extra),amended=True)

    def plans(self):
        old=dict(path='/old',bytes=1,sha256='old')
        amended=dict(path='/amendment',bytes=2,sha256='new')
        manifest=dict(format='pluto-paired-corpus-training-v1',replacement=dict(from_='Exeunt'))
        manifest['replacement']={'from':'Exeunt','to':'Nuveth','case_sensitive':True}
        amendment=dict(format='pluto-paired-lowercase-amendment-v1',complete=True,old_manifest=old,
                       replacement_rules=[{'source':'Exeunt','target':'Nuveth'},{'source':'exeunt','target':'nuveth'}])
        plan=dict(format=contract.AMENDED[0],complete=True,candidate_pairs=contract.PAIRS,
                  old_manifest=old,amendment=amended,provenance=[],sources=[])
        return plan,{'/old':manifest,'/amendment':amendment}

    def test_both_amended_suite_formats_bind_the_two_substitutions(self):
        for fmt in contract.AMENDED:
            plan,objects=self.plans(); plan['format']=fmt
            register=mock.Mock()
            self.assertTrue(contract.validate_plan(plan,register,objects.__getitem__))
            self.assertEqual(register.call_count,2)
            self.assertEqual(contract.experiment_identity(plan),plan['amendment'])

    def test_policy_cannot_revert_to_uppercase_only_or_wrong_original_manifest(self):
        plan,objects=self.plans()
        for mutate in (lambda obj:obj['/amendment']['replacement_rules'].pop(),
                       lambda obj:obj['/amendment']['old_manifest'].update(sha256='wrong'),
                       lambda obj:obj['/old']['replacement'].update(case_sensitive=False)):
            bad=copy.deepcopy(objects); mutate(bad)
            with self.assertRaises(ValueError): contract.validate_plan(plan,mock.Mock(),bad.__getitem__)

    def test_registers_nested_sources_and_provenance(self):
        plan,objects=self.plans()
        extra=dict(path='/source',bytes=3,sha256='src'); plan['sources']=[extra]
        register=mock.Mock()
        contract.validate_plan(plan,register,objects.__getitem__)
        register.assert_any_call('/source',extra)


if __name__=='__main__':
    unittest.main()
