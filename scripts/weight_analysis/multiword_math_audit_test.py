"""Focused audit-boundary/numerical tests; these do not replace real trace audit."""

import copy
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

import numpy as np

from . import multiword_math_audit as audit


def selection():
    return dict(words=[dict(word='grandam', word_steps=[365, 366], boundary_step=367),
                       dict(word='corse', word_steps=[700, 701], boundary_step=702),
                       dict(word='Exeunt', word_steps=[1340, 1341, 1342], boundary_step=1343)],
                events=[dict(step=step, token_id=step) for step in
                        [365, 366, 367, 700, 701, 702, 1340, 1341, 1342, 1343]], inputs=[])


def runner_plan():
    data = selection()
    word_steps = {step for word in data['words'] for step in word['word_steps']}
    return dict(schema_version=1, inputs=[],
                selections=[dict(word=word['word'], first=word['word_steps'][0],
                                 end=word['boundary_step'], boundary_step=word['boundary_step'])
                            for word in data['words']],
                runs=[dict(step=event['step'], event=event,
                           interventions=event['step'] in word_steps) for event in data['events']])


class FrozenSelectionTest(unittest.TestCase):
    def test_seven_word_steps_and_three_boundaries_remain_distinct(self):
        steps, words, targets = audit.selection_layout(selection())
        self.assertEqual(len(steps), 10)
        self.assertEqual(len(words), 7)
        self.assertEqual(set(steps)-words, {367, 702, 1343})
        self.assertEqual(targets[1342], 1342)

    def test_malformed_word_and_event_sets_rejected(self):
        mutations = [lambda p: p['words'][0].update(word_steps=[365]),
                     lambda p: p['words'][0].update(word_steps=[365, 367]),
                     lambda p: p['words'][0].update(word_steps=[False, 1]),
                     lambda p: p['words'][0].update(boundary_step=368),
                     lambda p: p['words'][1].update(word='GRANDAM'),
                     lambda p: p['words'].append(copy.deepcopy(p['words'][0])),
                     lambda p: p['events'].pop(),
                     lambda p: p['events'][0].update(step=366),
                     lambda p: p['events'][0].update(token_id=-1),
                     lambda p: p['events'][0].update(token_id=True)]
        for mutate in mutations:
            plan = selection(); mutate(plan)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                audit.selection_layout(plan)

    def test_native_runner_half_open_selection_schema(self):
        self.assertEqual(audit.normalize_runner_plan(runner_plan()), selection())

    def test_native_runner_cannot_drop_or_move_interventions(self):
        for mutate in (lambda p: p['runs'][0].update(interventions=False),
                       lambda p: p['runs'][2].update(interventions=True),
                       lambda p: p['runs'][0].update(interventions=1),
                       lambda p: p['runs'].reverse(),
                       lambda p: p['selections'][0].update(first=365.0),
                       lambda p: p.update(schema_version=2)):
            plan = runner_plan(); mutate(plan)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                audit.normalize_runner_plan(plan)

    def test_prompt_length_and_sliding_window_derived_independently(self):
        tokens = list(range(2200))
        for initial, step in [(6, 365), (6, 1018), (6, 1019), (6, 1340), (19, 1343)]:
            end = initial+step; start = max(0, end-1024)
            event = dict(step=step, absolute_token_index=end, context_start=start,
                         context_length=end-start, output_row=end-start-1)
            self.assertEqual(audit.generation_context(tokens, initial, step, event), tokens[start:end])
            for key in ('context_start', 'context_length', 'output_row', 'absolute_token_index'):
                corrupt = dict(event); corrupt[key] += 1
                with self.subTest(initial=initial, step=step, key=key), self.assertRaises(ValueError):
                    audit.generation_context(tokens, initial, step, corrupt)

    def test_zero_or_past_end_context_is_rejected(self):
        for initial, step, limit in [(0, 0, 1024), (1, -1, 1024), (1, 9, 1024), (1, 0, 0)]:
            with self.assertRaises(ValueError):
                audit.generation_context(list(range(10)), initial, step, {}, limit)


class PrimitiveMathTest(unittest.TestCase):
    def test_bf16_round_to_nearest_ties_to_even_including_negative(self):
        bits = np.array([0x3f808000, 0x3f818000, 0xbf808000, 0xbf818000,
                         0x3f807fff, 0x3f808001, 0x00000000, 0x80000000], dtype=np.uint32)
        expected = np.array([0x3f800000, 0x3f820000, 0xbf800000, 0xbf820000,
                             0x3f800000, 0x3f810000, 0, 0x80000000], dtype=np.uint32)
        actual = audit.quantized(bits.view(np.float32)).astype(np.float32).view(np.uint32)
        np.testing.assert_array_equal(actual, expected)

    def test_sampler_uses_fp32_subtraction_and_sequential_double_sum(self):
        row = np.array([0.7654321, -0.123456789, 0.5555556, -1.1], dtype=np.float32)
        maximum = float(row.max())
        # Independent scalar reconstruction with explicit binary32 pack/unpack.
        shifted = [struct.unpack('<f', struct.pack('<f', float(v)-maximum))[0] for v in row]
        self.assertTrue(any(s != float(v)-maximum for s, v in zip(shifted, row)))
        weights = [math.exp(s/0.8) for s in shifted]
        total = 0.0
        for value in weights:
            total += value
        probabilities = [value/total for value in weights]
        cdf = []; current = 0.0
        for value in probabilities:
            current += value; cdf.append(current)
        cdf[-1] = 1.0
        for target in range(4):
            p, lower, upper, chosen = audit.sampler(row, 0.8, 0.6, target)
            self.assertEqual(p, probabilities[target])
            self.assertEqual((lower, upper), (cdf[target-1] if target else 0, cdf[target]))
            self.assertEqual(chosen, next(i for i, v in enumerate(cdf) if v >= 0.6))

    def test_sampler_exact_cdf_boundary_belongs_to_previous_token(self):
        self.assertEqual(audit.sampler([0, 0], 1, 0.5, 1), (0.5, 0.5, 1.0, 0))
        self.assertEqual(audit.sampler([0, 0], 1, 0, 0), (0.5, 0.0, 0.5, 0))
        self.assertEqual(audit.sampler([0, 0], 1, math.nextafter(0.5, 1), 1)[-1], 1)

    def test_readout_checks_ranking_ties_logits_probability_cdf_and_draw(self):
        row = np.array([1, 1, 0, -2], dtype=np.float32)
        p, lower, upper, selected = audit.sampler(row, 0.8, 0.7, 1)
        exps = [math.exp(float(v)-1) for v in row]
        denominator = math.fsum(exps)
        entries = [dict(id=i, rank=i+1, logit=float(v), probability=exps[i]/denominator)
                   for i, v in enumerate(row)]
        saved = dict(target=entries[1], winner=entries[0], runner_up=entries[1], top5=entries,
                     target_probability_at_generation_temperature=p,
                     target_cdf_lower=lower, target_cdf_upper=upper, same_uniform_selected_id=selected)
        with mock.patch.object(audit, 'V', 4):
            self.assertEqual(audit._check_readout(row, saved, 1, 0.8, 0.7, metrics=audit.new_metrics()), p)
            mutations = [lambda x: x.update(target_cdf_upper=0.1),
                         lambda x: x.update(same_uniform_selected_id=3),
                         lambda x: x.update(target_probability_at_generation_temperature=0.1),
                         lambda x: x['target'].update(rank=1),
                         lambda x: x['target'].update(logit=4),
                         lambda x: x['target'].update(probability=0.1),
                         lambda x: x['top5'].reverse()]
            for mutate in mutations:
                bad = copy.deepcopy(saved); mutate(bad)
                with self.subTest(mutation=mutate), self.assertRaises(AssertionError):
                    audit._check_readout(row, bad, 1, 0.8, 0.7, metrics=audit.new_metrics())


class InputIntegrityTest(unittest.TestCase):
    def test_identity_recheck_detects_same_size_input_change(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'input'; path.write_bytes(b'abc')
            original = audit.identity(path); path.write_bytes(b'xyz')
            with self.assertRaises(AssertionError):
                audit._remember(path, original, records={})

    def test_raw_array_type_size_finiteness_and_directory(self):
        with tempfile.TemporaryDirectory() as root:
            native = Path(root)/'trace'; native.mkdir()
            path = native/'array'; path.write_bytes(np.array([0x3f80, 0x4000], dtype='<u2').tobytes())
            spec = dict(file='array', dtype='bf16', shape=[1, 2])
            np.testing.assert_array_equal(audit._raw(native, spec, records={}), [[1.0, 2.0]])
            with self.assertRaises(AssertionError):
                audit._raw(native, dict(spec, shape=[3]), records={})
            with self.assertRaises(AssertionError):
                audit._raw(native, dict(spec, file='../array'), records={})
            path.write_bytes(np.array([1, np.nan], dtype='<f4').tobytes())
            with self.assertRaises(AssertionError):
                audit._raw(native, dict(spec, dtype='float32'), records={})

    def test_existing_output_refused_before_loading_inputs(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root)/'output'; output.write_text('preserve')
            with self.assertRaises(FileExistsError):
                audit.run(root, '/missing-generation', '/missing-result', output=output)
            self.assertEqual(output.read_text(), 'preserve')

    def test_incomplete_runner_result_rejected_before_math(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            plan = runner_plan(); plan.update(generation_directory=str(root/'gen'), output_directory=str(root))
            (root/'plan.json').write_text(json.dumps(plan))
            (root/'result.json').write_text(json.dumps(dict(complete=False, all_inputs_unchanged=True)))
            with self.assertRaises(AssertionError):
                audit.run(root, root/'gen', root/'generation_result.json')
            self.assertFalse((root/'multiword_math_independent_audit.json').exists())


if __name__ == '__main__':
    unittest.main()
