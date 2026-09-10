"""Small independent interval oracles; no GPU or learned-model evidence."""
import unittest
from .paired_training_exposure import exposure,case_coverage


class TrainingExposureTest(unittest.TestCase):
    def test_target_shift_and_clipped_words_against_scalar_oracle(self):
        starts=[0,2,6,2];positions=[0,3,7]
        actual=exposure(starts,positions,token_count=12,context_length=4)
        piece=[];complete=[];partial=[]
        for p in positions:
            hits=[[p+i in range(s+1,s+5) for i in range(3)] for s in starts]
            piece.append([sum(h[i] for h in hits) for i in range(3)])
            complete.append(sum(all(h) for h in hits))
            partial.append(sum(any(h) and not all(h) for h in hits))
        self.assertEqual(actual,dict(piece_target_counts=piece,complete_target_counts=complete,
                                     partial_target_window_counts=partial))
        self.assertEqual(actual['complete_target_counts'][0],0)
        self.assertEqual(actual['partial_target_window_counts'][0],1)

    def test_zero_steps_and_no_words(self):
        self.assertEqual(exposure([], [1],token_count=10,context_length=4),
                         dict(piece_target_counts=[[0,0,0]],complete_target_counts=[0],partial_target_window_counts=[0]))
        self.assertEqual(exposure([0], [],token_count=10,context_length=4),
                         dict(piece_target_counts=[],complete_target_counts=[],partial_target_window_counts=[]))

    def test_complete_target_does_not_require_last_piece_in_input(self):
        result=exposure([0], [2],token_count=8,context_length=4)
        self.assertEqual(result['complete_target_counts'],[1])
        self.assertEqual(result['piece_target_counts'],[[1,1,1]])

    def test_prefix_segment_versus_identical_causal_prefix(self):
        result=case_coverage([0,1,2,3,4],prefix_start=1,word_start=4,context_length=6)
        self.assertEqual(result,dict(complete_target_presentations=4,
            prefix_segment_and_targets_presentations=2,identical_case_prefix_and_local_positions=1))

    def test_chunking_and_repeated_windows_count_presentations_not_unique_windows(self):
        result=exposure([0]*257,[1],token_count=10,context_length=4)
        self.assertEqual(result['complete_target_counts'],[257])
        self.assertEqual(result['piece_target_counts'],[[257]*3])

    def test_invalid_positions_contexts_and_prefix_geometry(self):
        for starts,positions,total,context in [([True],[1],10,4),([1.0],[1],10,4),
                ([6],[1],10,4),([-1],[1],10,4),([0],[8],10,4),([0],[2,1],10,4),
                ([0],[1,2],10,4),([0],[1],10,2),([0],[1],4,4)]:
            with self.subTest(values=(starts,positions,total,context)),self.assertRaises(ValueError):
                exposure(starts,positions,token_count=total,context_length=context)
        for prefix,word in [(1,1),(-1,2),(True,2)]:
            with self.assertRaises(ValueError):case_coverage([0],prefix_start=prefix,word_start=word,context_length=4)


if __name__=='__main__':unittest.main()
