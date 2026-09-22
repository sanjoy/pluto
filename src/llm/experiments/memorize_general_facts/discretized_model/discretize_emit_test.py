#!/usr/bin/env python3
"""CPU-only emitter invariants and separation of model versus evidence."""

import ast
import copy
import hashlib
import itertools
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

# Keep direct test execution and discovery independent of the current directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from src.llm.experiments.memorize_general_facts.discretized_model import discretize_emit


def fixture():
    return {
        "schema": 1, "width": 2, "layers": 1, "vocab_size": 4,
        "eos_token": 3, "prompt_tokens": 1,
        "vocabulary": [{"original_id": i+10, "hex": x.hex()}
                       for i, x in enumerate([b"A", b"B", b"C", b"<eos>"])],
        "samples": [{"tokens": [0, 1]}, {"tokens": [1]}],
        "states": [{"id": i, "stage": (i-4)//4, "bits": [i, 0]}
                   for i in range(4, 16)],
        "entry": [[0, 0, 4], [1, 0, 7], [1, 1, 5], [2, 1, 6]],
        "attention": [[[[4], 8], [[4, 5], 9], [[4, 6], 10], [[7], 11]]],
        "mlp": [[[8, 12], [9, 13], [10, 14], [11, 15]]],
        "language_modeling_head": [[12, 1], [13, 3], [14, 0], [15, 3]],
        "stats": {},
    }


class TokenNamesTest(unittest.TestCase):
    def assert_identifier(self, name):
        self.assertRegex(name, r"\Ak[A-Za-z0-9_]+_[0-9]+\Z")
        self.assertTrue(name.isascii())
        self.assertNotIn("__", name)
        stem, compact_id = name[1:].rsplit("_", 1)
        self.assertLessEqual(len(stem), 80)
        self.assertTrue(compact_id.isdecimal())

    def test_leading_spaces_case_and_eos_are_explicit(self):
        self.assertEqual(discretize_emit._token_name(b" France", 123, 999), "kSpace_France_123")
        self.assertEqual(discretize_emit._token_name(b"France", 123, 999), "kFrance_123")
        self.assertEqual(discretize_emit._token_name(b"The", 456, 999), "kThe_456")
        self.assertEqual(discretize_emit._token_name(b"the", 456, 999), "kthe_456")
        self.assertEqual(discretize_emit._token_name(b"<|endoftext|>", 4474, 4474), "kEos_4474")
        self.assertEqual(discretize_emit._token_name(b"unusual EOS bytes", 2, 2), "kEos_2")

    def test_id_suffix_prevents_mnemonic_and_duplicate_byte_collisions(self):
        tokens = [b" ", b"Space", b"_", b"Underscore", b"/", b"Slash",
                  b" a", b"Space_a", b"a-b", b"a_b", b"a b", b"a", b"a"]
        names = [discretize_emit._token_name(token, index, 999)
                 for index, token in enumerate(tokens)]
        self.assertEqual(len(names), len(set(names)))
        self.assertIn("Underscore", names[2])
        for name in names:
            self.assert_identifier(name)

    def test_every_byte_and_source_injection_text_remain_safe_identifiers(self):
        tokens = [bytes([byte]) for byte in range(256)]
        tokens.extend([b'*/\n#error injected\n//', b'"; namespace bad {',
                       b"\\\r\n", b"__reserved", bytes(range(256)),
                       "\u00e9\u00e8\U0001f642".encode("utf-8")])
        for index, token in enumerate(tokens):
            with self.subTest(token=token):
                name = discretize_emit._token_name(token, index, 999)
                self.assert_identifier(name)
                self.assertEqual(name, discretize_emit._token_name(token, index, 999))

    def test_cpp_keywords_numbers_and_long_tokens_are_legal_and_distinct(self):
        for index, token in enumerate([b"class", b"int", b"namespace", b"0", b"123abc"]):
            name = discretize_emit._token_name(token, index, 999)
            self.assertEqual(name, "k" + token.decode("ascii") + "_" + str(index))
            self.assert_identifier(name)
        common = b"A" * 200
        names = [discretize_emit._token_name(token, index, 999)
                 for index, token in enumerate([common + b"B", common + b"C", b"_" * 100])]
        self.assertEqual(len(names), len(set(names)))
        for name in names:
            self.assert_identifier(name)


class EmitModelTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = fixture()

    def emit(self, name="generated", model=None, **options):
        destination = self.root / name
        manifest = discretize_emit.emit_model(self.model if model is None else model, destination, **options)
        return destination, manifest

    def test_compact_emission_without_scripts_on_import_path(self):
        # Importing the experiment module directly must also resolve the compact
        # helpers, without a driver or test discovery supplying their directory.
        model_path = self.root / "model.json"
        model_path.write_text(json.dumps(self.model))
        output = self.root / "compact"
        command = """
import json
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
from src.llm.experiments.memorize_general_facts.discretized_model.discretize_emit import emit_model
emit_model(json.loads(Path(sys.argv[2]).read_text()), Path(sys.argv[3]),
           compact_transitions=True)
"""
        result = subprocess.run(
            [sys.executable, "-B", "-c", command,
             str(Path(__file__).resolve().parents[5]), str(model_path), str(output)],
            cwd=self.root, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads((output / "manifest.json").read_text())[
            "transition_representation"], "control_flow")

    def test_plain_split_sources_reproducible_under_input_table_order(self):
        first, manifest = self.emit("first")
        reordered = copy.deepcopy(self.model)
        for field in ("states", "entry", "language_modeling_head"):
            reordered[field].reverse()
        reordered["attention"][0].reverse()
        reordered["mlp"][0].reverse()
        second, repeated = self.emit("second", reordered)
        self.assertEqual(manifest, repeated)
        self.assertEqual({p.name: p.read_bytes() for p in first.iterdir()},
                         {p.name: p.read_bytes() for p in second.iterdir()})
        self.assertTrue((first / "attention_0.cc").exists())
        self.assertTrue((first / "mlp_0.cc").exists())
        build = (first / "BUILD.bazel").read_text()
        self.assertNotIn("genrule", build)
        self.assertNotIn("cuda", build.lower())
        self.assertIn(f'srcs = ["{discretize_emit.PACKAGE}:main.cc"]', build)
        self.assertNotIn(f'"{discretize_emit.PACKAGE}:main"', build)
        self.assertNotIn(str(self.root), "".join(p.read_text() for p in first.iterdir()))
        for name, digest in manifest["files"].items():
            self.assertEqual(hashlib.sha256((first / name).read_bytes()).hexdigest(), digest)

    def test_plain_rows_are_private_and_every_boundary_exposes_an_interface(self):
        destination, _ = self.emit()
        header = (destination / "tables.h").read_text()
        self.assertNotIn("EntryRow", header)
        self.assertNotIn("StateRow", header)
        self.assertNotIn("AttentionRow", header)
        self.assertNotIn("GeneratedEntry()", header)
        self.assertIn("PositionEmbedding& GeneratedPositionEmbedding();", header)
        self.assertIn("CausalAttention& GeneratedAttention0();", header)
        self.assertIn("Map& GeneratedMlp0();", header)
        self.assertIn("Map& GeneratedLanguageModelingHead();", header)
        model = (destination / "model.cc").read_text()
        self.assertIn("const DiscreteModel& GeneratedModel()", model)
        self.assertIn("internal::GeneratedLanguageModelingHead(), internal::GeneratedPositionEmbedding()", model)
        self.assertIn("{internal::GeneratedAttention0(), internal::GeneratedMlp0()}", model)
        self.assertNotIn("GeneratedEntry()", model)
        verification = (destination / "verification.cc").read_text()
        self.assertIn("model.prompt_token_count", verification)
        self.assertNotIn("model.prompt_tokens", verification)
        for name in ("entry.cc", "attention_0.cc", "mlp_0.cc", "language_modeling_head.cc"):
            text = (destination / name).read_text()
            self.assertRegex(text, r"namespace \{\nstruct (Entry|Attention|State)Row")
            self.assertIn("std::optional<DiscreteHiddenState>", text)
            self.assertIn("operator()", text)
            self.assertIn("static ", text)
            self.assertNotIn("TransitionResult", text)

    def test_public_header_exposes_only_the_model_factory(self):
        destination, _ = self.emit()
        header = (destination / "model.h").read_text()
        self.assertIn("namespace pluto::llm::discretized::gen", header)
        self.assertIn('[[gnu::visibility("default")]]', header)
        self.assertIn("const DiscreteModel& GeneratedModel();", header)
        self.assertEqual(re.findall(r"\bGenerated\w+\s*\(", header), ["GeneratedModel("])
        for private_header in ("tables.h", "vocabulary_tokens.h", "cli_support.h"):
            self.assertNotIn(private_header, header)
            self.assertIn("namespace pluto::llm::discretized::gen::internal",
                          (destination / private_header).read_text())

    def _compile_and_check_partial_functions(self, model, compact):
        """Run the actual emitted production TUs, independent of lookup strategy.

        A minimal header supplies only the public polymorphic interfaces and
        a standard span alias; no lookup algorithm is duplicated by this shim.
        This keeps the Python test CPU-only and independent of a Bazel cache.
        Repository C++ tests separately exercise the actual runtime itself.
        """
        directory, _ = self.emit(model=model, compact_transitions=compact)
        if model["layers"] == 0:
            self.assertNotIn("kTransformers", (directory / "model.cc").read_text())
        include = self.root / "include"
        header = include / discretize_emit.RUNTIME
        header.parent.mkdir(parents=True)
        header.write_text('''#pragma once
#include <cstddef>
#include <compare>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
namespace absl { template<class T> using Span = std::span<T>; }
namespace pluto::llm::discretized {
struct DiscreteHiddenState;
struct DiscreteToken {
  int value = 0;
  constexpr explicit operator DiscreteHiddenState() const;
  constexpr auto operator<=>(const DiscreteToken&) const = default;
};
struct DiscreteHiddenState {
  int value = 0;
  constexpr explicit operator DiscreteToken() const { return DiscreteToken{value}; }
  constexpr auto operator<=>(const DiscreteHiddenState&) const = default;
};
constexpr DiscreteToken::operator DiscreteHiddenState() const {
  return DiscreteHiddenState{value};
}
struct VocabularyRow { int32_t original_id; std::string_view bytes; };
class CausalAttention {
 public:
  virtual ~CausalAttention() = default;
  virtual std::optional<DiscreteHiddenState> operator()(absl::Span<const DiscreteHiddenState>) = 0;
};
class Map {
 public:
  virtual ~Map() = default;
  virtual std::optional<DiscreteHiddenState> operator()(DiscreteHiddenState) = 0;
};
class PositionEmbedding {
 public:
  virtual ~PositionEmbedding() = default;
  virtual std::optional<DiscreteHiddenState> operator()(DiscreteToken, int32_t) = 0;
};
struct Transformer {
  CausalAttention& attention;
  Map& mlp;
};
struct DiscreteModel {
  uint32_t context_length;
  uint32_t prompt_token_count;
  DiscreteToken eos_token;
  absl::Span<const VocabularyRow> vocabulary;
  absl::Span<const Transformer> transformers;
  Map& language_modeling_head;
  PositionEmbedding& position_embedding;
};
}
''')
        checks = []

        def check(expression, expected):
            output = ("std::nullopt" if expected is None else
                      f"std::optional<DiscreteHiddenState>(DiscreteHiddenState{{{expected}}})")
            label = len(checks)
            checks.append(f'if (({expression}) != {output}) {{ '
                          f'std::cerr << "partial function check {label} failed\\n"; return 1; }}')

        entries = {(token, position): output for token, position, output in model["entry"]}
        for token in [-1, *range(model["vocab_size"] + 1), 2147483647]:
            for position in [-2147483648, -1, *range(5), 2147483647]:
                check(f"model.position_embedding(DiscreteToken{{{token}}}, {position})", entries.get((token, position)))
        pointwise = [(f"model.transformers[{block}].mlp", dict(rows))
                     for block, rows in enumerate(model["mlp"])]
        pointwise.append(("model.language_modeling_head", dict(model["language_modeling_head"])))
        for function, rows in pointwise:
            for state in [-2147483648, -1, *range(20), *rows, 2147483646, 2147483647]:
                check(f"{function}(DiscreteHiddenState{{{state}}})", rows.get(state))
        for block, rows in enumerate(model["attention"]):
            checks.append(f'''if (&model.transformers[{block}].attention != &gen::internal::GeneratedAttention{block}() ||
    &model.transformers[{block}].mlp != &gen::internal::GeneratedMlp{block}()) return 1;''')
            by_key = {tuple(prefix): output for prefix, output in rows}
            alphabet = sorted({-1, 0, 4, 5, 6, 7, 2147483647} |
                              {state for prefix, _ in rows for state in prefix})
            probes = {()} | set(by_key)
            for length in range(1, 4):
                probes.update(itertools.product(alphabet, repeat=length))
            for prefix in sorted(probes):
                literal = ", ".join(f"DiscreteHiddenState{{{state}}}" for state in prefix)
                check(f"model.transformers[{block}].attention(std::vector<DiscreteHiddenState>{{{literal}}})",
                      by_key.get(prefix))
        source = directory / "partial_functions_test.cc"
        source.write_text('''#include <iostream>
#include <vector>
#include "model.h"
#include "tables.h"
using namespace pluto::llm::discretized;
int main() {
  const DiscreteModel& model = gen::GeneratedModel();
''' + f'''  if (model.transformers.size() != {model["layers"]})
    return 1;
  if (&model.position_embedding != &gen::internal::GeneratedPositionEmbedding() ||
      &model.language_modeling_head != &gen::internal::GeneratedLanguageModelingHead())
    return 1;
''' + "\n".join(checks) + "\nreturn 0;\n}\n")
        sources = [directory / name for name in ("entry.cc", "model.cc", "language_modeling_head.cc", "vocabulary.cc")]
        sources += [directory / f"{kind}_{block}.cc" for kind in ("attention", "mlp")
                    for block in range(model["layers"])]
        executable = directory / "partial_functions_test"
        compiled = subprocess.run(
            ["c++", "-std=c++20", "-O1", "-Wall", "-Wextra", "-Werror",
             "-I", str(include), *map(str, sources), str(source), "-o", str(executable)],
            capture_output=True, text=True, timeout=60)
        self.assertEqual(compiled.returncode, 0, compiled.stderr)
        ran = subprocess.run([str(executable)], capture_output=True, text=True, timeout=10)
        self.assertEqual(ran.returncode, 0, ran.stderr)

        # A real shared-library build verifies symbol visibility, independently
        # of header privacy. Internal factories must not leak into the ABI.
        if shutil.which("nm") is not None:
            shared_library = directory / "libmodel.so"
            linked = subprocess.run(
                ["c++", "-std=c++20", "-O1", "-fPIC", "-shared",
                 "-fvisibility=hidden", "-fvisibility-inlines-hidden",
                 "-I", str(include), *map(str, sources), "-o", str(shared_library)],
                capture_output=True, text=True, timeout=60)
            self.assertEqual(linked.returncode, 0, linked.stderr)
            symbols = subprocess.run(
                ["nm", "-D", "--defined-only", "--demangle", str(shared_library)],
                capture_output=True, text=True, check=True, timeout=10)
            exported = [line.split(maxsplit=2)[-1] for line in symbols.stdout.splitlines()
                        if "pluto::llm::discretized::gen::" in line]
            self.assertEqual(exported, ["pluto::llm::discretized::gen::GeneratedModel()"])
            consumer = directory / "public_api_test.cc"
            consumer.write_text('''#include "model.h"
int main() {
  const auto& model = pluto::llm::discretized::gen::GeneratedModel();
  return model.context_length == 1024 ? 0 : 1;
}
''')
            public_executable = directory / "public_api_test"
            linked = subprocess.run(
                ["c++", "-std=c++20", "-I", str(include), str(consumer),
                 str(shared_library), "-o", str(public_executable)],
                capture_output=True, text=True, timeout=60)
            self.assertEqual(linked.returncode, 0, linked.stderr)
            ran = subprocess.run([str(public_executable)], capture_output=True, text=True, timeout=10)
            self.assertEqual(ran.returncode, 0, ran.stderr)

    @unittest.skipUnless(shutil.which("c++"), "C++ compiler unavailable")
    def test_plain_compiled_functions_match_exact_domains_and_supported_zero(self):
        # Unsorted inputs exercise sorting before the binary-search emission.
        self.model["entry"].reverse()
        self.model["attention"][0].reverse()
        self.model["mlp"][0].reverse()
        self.model["language_modeling_head"].reverse()
        self._compile_and_check_partial_functions(self.model, compact=False)

    @unittest.skipUnless(shutil.which("c++"), "C++ compiler unavailable")
    def test_compact_functions_expose_the_same_interface_and_partial_domains(self):
        self._compile_and_check_partial_functions(self.model, compact=True)

    @unittest.skipUnless(shutil.which("c++"), "C++ compiler unavailable")
    def test_plain_empty_attention_and_mlp_reject_every_input(self):
        self.model["attention"] = [[]]
        self.model["mlp"] = [[]]
        self._compile_and_check_partial_functions(self.model, compact=False)

    @unittest.skipUnless(shutil.which("c++"), "C++ compiler unavailable")
    def test_plain_zero_blocks_and_maximum_state_ids(self):
        self.model.update(layers=0,
                          states=[{"id": 2147483647, "stage": 0, "bits": [0, 0]}],
                          entry=[[0, 0, 2147483647]], attention=[], mlp=[],
                          language_modeling_head=[[2147483647, 0]], samples=[{"tokens": [0]}])
        self._compile_and_check_partial_functions(self.model, compact=False)

    @unittest.skipUnless(shutil.which("c++"), "C++ compiler unavailable")
    def test_compact_zero_blocks_and_maximum_state_ids(self):
        self.model.update(layers=0,
                          states=[{"id": 2147483647, "stage": 0, "bits": [0, 0]}],
                          entry=[[0, 0, 2147483647]], attention=[], mlp=[],
                          language_modeling_head=[[2147483647, 3]], samples=[{"tokens": [0]}])
        self._compile_and_check_partial_functions(self.model, compact=True)

    def test_expected_suffixes_never_enter_production_model_sources(self):
        first, _ = self.emit("first")
        alternate = copy.deepcopy(self.model)
        alternate["samples"] = [{"tokens": [0, 2]}, {"tokens": [1]}]
        second, _ = self.emit("second", alternate)
        production = ["model.h", "tables.h", "vocabulary_tokens.h", "model.cc", "entry.cc", "attention_0.cc",
                      "mlp_0.cc", "language_modeling_head.cc", "vocabulary.cc"]
        for name in production:
            self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
            text = (first / name).read_text()
            self.assertNotIn("kExpectedTokens", text)
            self.assertNotIn("sample", text.lower())
        self.assertNotEqual((first / "verification.cc").read_bytes(),
                            (second / "verification.cc").read_bytes())

    def test_named_vocabulary_constants_have_exact_ids_and_cover_all_token_references(self):
        destination, manifest = self.emit()
        header = (destination / "vocabulary_tokens.h").read_text()
        declarations = dict(re.findall(r"inline\s+constexpr\s+DiscreteToken\s+(\w+)\s*\{(\d+)\}\s*;", header))
        self.assertEqual(declarations, {"kA_0": "0", "kB_1": "1", "kC_2": "2", "kEos_3": "3"})
        self.assertIn(discretize_emit.NAMESPACE + "::gen::internal::vocab", header)
        self.assertIn("vocabulary_tokens.h", manifest["files"])
        names = ["kA_0", "kB_1", "kC_2", "kEos_3"]
        entry = (destination / "entry.cc").read_text()
        for token, position, state in self.model["entry"]:
            self.assertIn(f"{{vocab::{names[token]}, {position}, {{{state}}}}}", entry)
        language_modeling_head = (destination / "language_modeling_head.cc").read_text()
        for state, token in self.model["language_modeling_head"]:
            self.assertIn(f"{{{{{state}}}, static_cast<DiscreteHiddenState>(vocab::{names[token]})}}", language_modeling_head)
        self.assertIn("1024, 1, internal::vocab::kEos_3", (destination / "model.cc").read_text())
        for filename, array, expected in (
                ("prompt_encoder.cc", "kTokens", ["kA_0", "kA_0", "kB_1", "kB_1"]),
                ("verification.cc", "kExpectedTokens", ["kA_0", "kB_1", "kB_1"])):
            text = (destination / filename).read_text()
            body = re.search(r"const DiscreteToken " + array + r"\[\]\s*=\s*\{(.*?)\};", text, re.S)
            self.assertIsNotNone(body)
            values = [value.strip() for value in body.group(1).split(",") if value.strip()]
            self.assertEqual(values, ["vocab::" + name for name in expected])
        # Attention and MLP symbols remain numeric hidden-state IDs.
        self.assertNotIn("vocab::", (destination / "attention_0.cc").read_text())
        self.assertNotIn("vocab::", (destination / "mlp_0.cc").read_text())
        self.assertIn("{{8}, {12}}", (destination / "mlp_0.cc").read_text())

    def test_named_vocabulary_header_has_shared_explicit_build_dependencies(self):
        destination, _ = self.emit()
        build = ast.parse((destination / "BUILD.bazel").read_text())
        rules = {}
        for node in build.body:
            if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == "cc_library"):
                fields = {keyword.arg: ast.literal_eval(keyword.value)
                          for keyword in node.value.keywords}
                rules[fields["name"]] = fields
        self.assertEqual(rules["vocabulary_tokens"]["hdrs"], ["vocabulary_tokens.h"])
        self.assertEqual(rules["vocabulary_tokens"]["visibility"], ["//visibility:private"])
        self.assertEqual(rules["model"]["hdrs"], ["model.h"])
        self.assertIn("tables.h", rules["model"]["srcs"])
        self.assertEqual(rules["model"]["include_prefix"], "pluto/discretized/gen")
        self.assertEqual(rules["model"]["strip_include_prefix"], ".")
        self.assertIn("-fvisibility=hidden", rules["model"]["copts"])
        self.assertIn("-fvisibility-inlines-hidden", rules["model"]["copts"])
        self.assertEqual(rules["cli_support"]["hdrs"], ["cli_support.h"])
        for target in ("cli_support", "prompt_encoder", "verification"):
            self.assertEqual(rules[target]["visibility"], ["//visibility:private"])
        # These helpers have hidden symbols, so they must be linked into their
        # private consumers rather than exposed through shared-library ABIs.
        for target in ("prompt_encoder", "verification"):
            self.assertTrue(rules[target]["linkstatic"])
        self.assertFalse(rules["model"].get("linkstatic", False))
        for target in ("model", "prompt_encoder", "verification"):
            self.assertIn(":vocabulary_tokens", rules[target]["deps"])

    def test_duplicate_token_bytes_get_distinct_named_constants(self):
        # The duplicate is unused by the text encoder, avoiding its separate
        # rejection of ambiguous prefix tokenization.
        self.model["vocabulary"][2]["hex"] = self.model["vocabulary"][0]["hex"]
        destination, _ = self.emit()
        header = (destination / "vocabulary_tokens.h").read_text()
        self.assertIn("kA_0{0}", header)
        self.assertIn("kA_2{2}", header)

    def test_exact_history_keys_are_flat_integer_sequences_not_hashes(self):
        destination, _ = self.emit()
        text = (destination / "attention_0.cc").read_text()
        self.assertIn("const DiscreteHiddenState kKeys[]", text)
        self.assertIn("{1, 2, {9}}", text)
        self.assertIn("{3, 2, {10}}", text)
        self.assertNotIn("hash", text)
        self.assertNotIn("float", text)

    def test_arbitrary_bytes_are_escaped_without_utf8_roundtrip(self):
        self.model["vocabulary"][0]["hex"] = "0080ff225c0a"
        destination, _ = self.emit()
        self.assertIn(r'"\000\200\377\"\\\n", 6',
                      (destination / "vocabulary.cc").read_text())

    def test_readable_literal_preserves_all_byte_values_and_escape_boundaries(self):
        contents = bytes(range(256)) + b"\xffa012\x00a7\"\\\n"
        literal = discretize_emit._literal(contents)
        self.assertEqual(ast.literal_eval(literal).encode("latin1"), contents)
        self.assertEqual(discretize_emit._literal(b" The capital of France"),
                         '" The capital of France"')
        self.assertIn(r"\377a012", literal)

    def test_generated_sources_explain_strict_boundaries_and_safe_token_comments(self):
        self.model["vocabulary"][0]["hex"] = b'*/\n// injected\n\\'.hex()
        destination, _ = self.emit()
        entry = (destination / "entry.cc").read_text()
        self.assertIn("zero_based_position", entry)
        self.assertIn(r'// token "*/\n// injected\n\\"', entry)
        self.assertNotIn("\n// injected", entry)
        self.assertIn("complete causal state prefix", (destination / "attention_0.cc").read_text())
        self.assertIn("pointwise MLP boundary", (destination / "mlp_0.cc").read_text())
        self.assertIn("All attention and MLP boundaries remain separate", (destination / "model.cc").read_text())

    def test_optional_state_index_counts_real_positions_and_names_every_boundary(self):
        self.model["samples"].append({"tokens": [0, 1]})
        destination, manifest = self.emit(include_state_index=True)
        self.assertTrue(manifest["state_index_included"])
        index = (destination / "state_index.tsv").read_text()
        self.assertIn("not semantic labels", index)
        lines = [line for line in index.splitlines() if not line.startswith("#")]
        rows = {int(fields[0]): fields for fields in (line.split("\t") for line in lines[1:])}
        self.assertEqual(set(rows), set(range(4, 16)))
        self.assertEqual(rows[4][1:4], ["token_plus_position_embedding", "0004 0000", "2"])
        self.assertEqual(rows[9][1], "block_0.after_attention_residual")
        self.assertEqual(rows[13][1], "block_0.after_mlp_residual")
        self.assertEqual(rows[6][3], "0")
        self.assertEqual(json.loads(rows[6][4]), [])
        self.assertEqual(json.loads(rows[13][4]), [{"compact_ids": [0, 1], "text": "AB"}])
        self.assertEqual(sum(int(row[3]) for row in rows.values()), 5 * 3)
        self.assertNotIn("state_index.tsv", (destination / "BUILD.bazel").read_text())
        second, repeated = self.emit("repeat", include_state_index=True)
        self.assertEqual(index, (second / "state_index.tsv").read_text())
        self.assertEqual(manifest, repeated)

    def test_index_examples_are_capped_and_long_text_is_explicitly_truncated(self):
        self.model.update(layers=0, states=[{"id": 4, "stage": 0, "bits": [4, 0]}],
                          entry=[[0, 0, 4], [0, 1, 4], [1, 0, 4], [2, 0, 4]],
                          attention=[], mlp=[], language_modeling_head=[[4, 3]],
                          samples=[{"tokens": [0]}, {"tokens": [1]}, {"tokens": [2]}, {"tokens": [0, 0]}])
        self.model["vocabulary"][0]["hex"] = (b"A" * 200).hex()
        destination, _ = self.emit(include_state_index=True)
        fields = (destination / "state_index.tsv").read_text().splitlines()[-1].split("\t")
        self.assertEqual(fields[3], "5")
        examples = json.loads(fields[4])
        self.assertEqual(len(examples), 3)
        self.assertEqual(examples[0], {"compact_ids": [0], "text": "A" * 160,
                                       "text_truncated_after_bytes": 160})

    def test_index_is_opt_in_and_cannot_alter_model_tables(self):
        ordinary, _ = self.emit()
        inspected, _ = self.emit("inspected", include_state_index=True)
        self.assertFalse((ordinary / "state_index.tsv").exists())
        for file in ordinary.iterdir():
            if file.name != "manifest.json":
                self.assertEqual(file.read_bytes(), (inspected / file.name).read_bytes())

    def test_existing_output_is_never_modified(self):
        destination, _ = self.emit()
        before = {p.name: p.read_bytes() for p in destination.iterdir()}
        with self.assertRaises(FileExistsError):
            self.emit()
        self.assertEqual(before, {p.name: p.read_bytes() for p in destination.iterdir()})

    def test_state_membership_is_separate_and_exact(self):
        for row in self.model["states"]:
            row["members"] = [row["id"] * 10, row["id"] * 10 + 1]
            row["member_count"] = 2
        destination, _ = self.emit(include_state_index=True)
        index = (destination / "state_index.tsv").read_text()
        rows = [line.split("\t") for line in index.splitlines() if not line.startswith("#")][1:]
        self.assertTrue(all(row[-1] == "2" for row in rows))
        members = (destination / "state_members.tsv").read_text()
        self.assertIn("[40,41]", members)
        self.assertNotIn("state_members.tsv", (destination / "BUILD.bazel").read_text())

    def test_unknown_membership_is_not_fabricated(self):
        destination, _ = self.emit(include_state_index=True)
        self.assertIn("\tunknown\n", (destination / "state_index.tsv").read_text())
        self.assertFalse((destination / "state_members.tsv").exists())

    def test_publication_race_refuses_even_empty_directory(self):
        publish = discretize_emit._publish
        def competitor(staging, destination):
            destination.mkdir()
            publish(staging, destination)
        with mock.patch.object(discretize_emit, "_publish", side_effect=competitor):
            with self.assertRaises(FileExistsError):
                self.emit()
        self.assertEqual(list((self.root / "generated").iterdir()), [])
        self.assertEqual(list(self.root.glob(".generated.emit-*")), [])

    def test_bad_models_fail_before_staging(self):
        invalid = []
        for value in (0, True, -1):
            model = fixture()
            model["width"] = value
            invalid.append(model)
        model = fixture()
        model["entry"].append(model["entry"][0])
        invalid.append(model)
        model = fixture()
        model["attention"][0].append([[4], 9])
        invalid.append(model)
        model = fixture()
        model["attention"][0][1][0][0] = 8  # Wrong boundary.
        invalid.append(model)
        model = fixture()
        model["language_modeling_head"][0][1] = 4
        invalid.append(model)
        model = fixture()
        model["states"][0]["bits"][0] = 65536
        invalid.append(model)
        model = fixture()
        model["vocabulary"][0]["hex"] = "FF"
        invalid.append(model)
        # Every symbolic state is stored in an int, including original-member
        # metadata and transition references; no uint32-only values may leak out.
        for value in (2**31, 2**32 - 1):
            model = fixture()
            model["states"][0]["id"] = value
            invalid.append(model)
            model = fixture()
            model["states"][0].update(members=[value], member_count=1)
            invalid.append(model)
            model = fixture()
            model["entry"][0][2] = value
            invalid.append(model)
        for model in invalid:
            with self.subTest(model=model), mock.patch.object(discretize_emit.tempfile, "TemporaryDirectory") as staging:
                with self.assertRaises(ValueError):
                    self.emit(model=model)
                staging.assert_not_called()
        self.assertFalse((self.root / "generated").exists())

    def test_ambiguous_text_encoder_is_rejected_without_changing_token_semantics(self):
        self.model["vocabulary"][2]["hex"] = "4142"  # C decodes to AB.
        self.model["samples"].append({"tokens": [2]})
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            self.emit()
        self.assertFalse((self.root / "generated").exists())


if __name__ == "__main__":
    unittest.main()
