#!/usr/bin/env python3
"""Independent, synthetic checks of exact final-prediction artifact coverage."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from verify_predictions import TSV_HEADER, _corpus_lines, main, verify_predictions


class VerifyPredictionsTest(unittest.TestCase):
    def setUp(self):
        self.rows = [[1, 2, 3, 4], [5, 6, 7], [8, 9]]
        # Two-token prompts; the third sample predicts EOS only. The indices
        # here are written out, independently of the verifier's range logic.
        self.records = [
            [1, 2, 3, 3, 0.1],
            [1, 3, 4, 4, 0.2],
            [1, 4, 0, 0, 0.3],
            [2, 2, 7, 7, 0.4],
            [2, 3, 0, 0, 0.5],
            [3, 2, 0, 0, 0.6],
        ]

    def verify(self, records=None, *, rows=None, header=TSV_HEADER, **options):
        if records is None:
            records = self.records
        if rows is None:
            rows = self.rows
        text = "\t".join(header) + "\n"
        text += "".join("\t".join(map(str, record)) + "\n" for record in records)
        arguments = dict(
            eos_id=0,
            vocabulary_size=10,
            prompt_tokens=2,
            context_length=4,
            expected_samples=3,
        )
        arguments.update(options)
        return verify_predictions(rows, io.StringIO(text), **arguments)

    def test_exact_coverage_includes_eos_and_excludes_prompt_and_padding(self):
        result = self.verify()
        self.assertEqual(result.targets, 6)
        self.assertEqual(result.sentences, 3)
        self.assertEqual(result.exact_sentences, 3)
        self.assertEqual(result.errors, 0)
        self.assertTrue(result.summary()["success"])
        self.assertEqual(result.summary()["accuracy"], 1)
        self.assertAlmostEqual(result.mean_loss, 0.35)

    def test_errors_and_exact_sentences_are_integer_counts(self):
        self.records[0][3] = 4
        self.records[2][3] = 1  # An incorrect EOS counts too.
        result = self.verify()
        self.assertEqual(result.errors, 2)
        self.assertEqual(result.exact_sentences, 2)
        self.assertEqual(result.targets, 6)
        self.assertFalse(result.summary()["success"])
        self.assertAlmostEqual(result.summary()["accuracy"], 4 / 6)

    def test_file_order_is_irrelevant(self):
        expected = self.verify()
        self.assertEqual(self.verify(list(reversed(self.records))), expected)

    def test_duplicate_rows_cannot_replace_missing_rows(self):
        records = self.records[:-1] + [self.records[0]]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.verify(records)

    def test_missing_target_and_entire_missing_sentence_are_errors(self):
        for records in [self.records[:-1], self.records[3:]]:
            with self.subTest(records=records):
                with self.assertRaisesRegex(ValueError, "missing"):
                    self.verify(records)

    def test_incorrect_declared_target_is_not_accepted_as_success(self):
        self.records[0][2] = 4
        self.records[0][3] = 4
        with self.assertRaisesRegex(ValueError, "independent tokenization requires 3"):
            self.verify()

    def test_prompt_padding_and_unknown_lines_are_extraneous(self):
        for extra in [
            [1, 1, 2, 2, 0],  # Inside prompt.
            [2, 4, 0, 0, 0],  # Padding after this sample's EOS.
            [4, 2, 0, 0, 0],  # Nonexistent sentence.
            [0, 2, 3, 3, 0],  # Line numbers are one-based.
        ]:
            with self.subTest(extra=extra):
                with self.assertRaisesRegex(ValueError, "unexpected"):
                    self.verify(self.records + [extra])

    def test_one_based_target_positions_are_rejected(self):
        shifted = [
            [line, pos + 1, target, pred, loss]
            for line, pos, target, pred, loss in self.records
        ]
        with self.assertRaises(ValueError):
            self.verify(shifted)

    def test_missing_eos_is_rejected(self):
        records = [record for record in self.records if record[2] != 0]
        with self.assertRaisesRegex(ValueError, "missing 3 scored targets"):
            self.verify(records)

    def test_nonfinite_loss_is_rejected(self):
        for loss in [math.nan, math.inf, -math.inf, "1e9999"]:
            with self.subTest(loss=loss):
                self.records[0][4] = loss
                with self.assertRaisesRegex(ValueError, "finite"):
                    self.verify()

    def test_noninteger_indices_and_predictions_are_rejected(self):
        for field in range(4):
            for value in ["1.0", "-1", "+1", " 1", "nan", ""]:
                with self.subTest(field=field, value=value):
                    records = [record.copy() for record in self.records]
                    records[0][field] = value
                    with self.assertRaisesRegex(ValueError, "unsigned integer"):
                        self.verify(records)

    def test_padded_vocabulary_predictions_cannot_count_as_real_tokens(self):
        self.records[0][3] = 10
        with self.assertRaisesRegex(ValueError, "logical vocabulary"):
            self.verify()

    def test_schema_must_match_exactly(self):
        for header in [TSV_HEADER[:-1], ("line",) + TSV_HEADER[1:], TSV_HEADER[::-1]]:
            with self.subTest(header=header):
                with self.assertRaisesRegex(ValueError, "header"):
                    self.verify(header=header)
        with self.assertRaisesRegex(ValueError, "five fields"):
            self.verify(self.records + [[]])
        with self.assertRaisesRegex(ValueError, "five fields"):
            self.verify(self.records + [[1, 2, 3, 3]])

    def test_empty_or_incomplete_corpus_cannot_reduce_the_accuracy_denominator(self):
        for rows in [[], self.rows[:-1], self.rows + [[1, 2]]]:
            with self.subTest(rows=rows):
                with self.assertRaisesRegex(ValueError, "exactly 3"):
                    self.verify(rows=rows)

    def test_invalid_sample_lengths_and_vocabulary_ids_fail(self):
        for sample in [[], [1], [1, 2, 3, 4, 5], [1, -1], [1, 10], [True, 2]]:
            with self.subTest(sample=sample):
                with self.assertRaises(ValueError):
                    self.verify(rows=[sample] + self.rows[1:])

    def test_invalid_scoring_configuration_fails(self):
        for options in [
            {"eos_id": -1},
            {"eos_id": 10},
            {"vocabulary_size": 0},
            {"prompt_tokens": 0},
            {"prompt_tokens": 5},
            {"context_length": 0},
            {"expected_samples": 0},
        ]:
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    self.verify(**options)

    def test_context_32_scores_a_26_token_sentence_in_original_gpt2_ids(self):
        sentence = list(range(5000, 5026))
        records = [[1, position, sentence[position], sentence[position], 0.1]
                   for position in range(5, 26)]
        records.append([1, 26, 50256, 50256, 0.1])
        result = self.verify(records, rows=[sentence], eos_id=50256,
                             vocabulary_size=50257, prompt_tokens=5,
                             context_length=32, expected_samples=1)
        self.assertEqual(result.targets, 22)
        self.assertEqual(result.errors, 0)
        self.assertEqual(result.predictions[-1].target_id, 50256)

    def test_context_32_rejects_a_33_token_sentence(self):
        with self.assertRaisesRegex(ValueError, "invalid token count"):
            self.verify([], rows=[list(range(33))], eos_id=50256,
                        vocabulary_size=50257, prompt_tokens=5,
                        context_length=32, expected_samples=1)

    def test_line_splitting_matches_native_lf_and_crlf_semantics(self):
        self.assertEqual(_corpus_lines(b"one\r\ntwo\r\n"), ["one", "two"])
        self.assertEqual(_corpus_lines("one\u2028two\n".encode()), ["one\u2028two"])
        for corpus in [b"", b"\n", b"one\n\n", b"one\n \t\n"]:
            with self.subTest(corpus=corpus):
                with self.assertRaisesRegex(ValueError, "nonempty"):
                    _corpus_lines(corpus)


class VerifyPredictionsCliTest(unittest.TestCase):
    arguments = ["--corpus=corpus.txt", "--tokenizer=tokenizer.json",
                 "--predictions=final_predictions.tsv"]

    def test_context_must_cover_the_five_token_prompt(self):
        for context in ("-1", "0", "4"):
            with self.subTest(context=context), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main(self.arguments + [f"--context_length={context}"])
                self.assertEqual(error.exception.code, 2)

    def test_selected_and_historical_context_are_recorded(self):
        # Both compact and full-vocabulary native audits retain original IDs.
        tokenizer = Mock()
        tokenizer.encode.return_value.ids = [5000, 5001, 5002, 5003, 5004]
        tokenizer.token_to_id.return_value = 50256
        tokenizer.get_vocab_size.return_value = 50257
        factory = Mock()
        factory.from_str.return_value = tokenizer
        predictions = "\t".join(TSV_HEADER) + "\n"
        predictions += "".join(f"{line}\t5\t50256\t50256\t0.1\n"
                               for line in range(1, 1025))
        for options, expected in (([], 1024), (["--context_length=32"], 32)):
            with self.subTest(context=expected):
                output = io.StringIO()
                with patch.dict("sys.modules", {"tokenizers": SimpleNamespace(Tokenizer=factory)}), \
                     patch.object(Path, "read_bytes", side_effect=[
                         b"fact\n" * 1024, b"{}", predictions.encode(),
                     ]), redirect_stdout(output):
                    self.assertEqual(main(self.arguments + options), 0)
                result = json.loads(output.getvalue())
                self.assertEqual(result["context_length"], expected)
                self.assertEqual(result["targets"], 1024)
                self.assertEqual(result["errors"], 0)


if __name__ == "__main__":
    unittest.main()
