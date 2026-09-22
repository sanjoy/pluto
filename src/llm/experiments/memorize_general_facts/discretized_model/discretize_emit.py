#!/usr/bin/env python3
"""Emit a standalone, exact integer network as ordinary split C++ sources.

Production transitions cannot access the independently linked verifier's expected
tokens. The optional text encoder contains input-prefix encodings only. Output
publication is atomic, refuses existing destinations, and has no absolute paths.
Every boundary exposes a pure function. The plain backend keeps its sorted data
private to each translation unit; compact generation replaces these wrappers
with equivalent control flow without changing the runtime interface.
"""

import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


RUNTIME = "src/llm/experiments/memorize_general_facts/discretized_model/runtime.h"
PACKAGE = "//src/llm/experiments/memorize_general_facts/discretized_model"
NAMESPACE = "pluto::llm::discretized"


def _integer(value, label, minimum=0, maximum=2**32 - 1):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{label} must be an integer in [{minimum}, {maximum}]")
    return value


def _validate(model):
    if type(model) is not dict or model.get("schema") != 1:
        raise ValueError("expected integer model schema 1")
    layers = _integer(model["layers"], "layers", maximum=1024)
    width = _integer(model["width"], "width", minimum=1)
    vocabulary_size = _integer(model["vocab_size"], "vocab_size", minimum=1, maximum=2**31-1)
    prompt = _integer(model["prompt_tokens"], "prompt_tokens", minimum=1, maximum=1024)
    _integer(model["eos_token"], "eos_token", maximum=vocabulary_size-1)
    vocabulary = model["vocabulary"]
    if len(vocabulary) != vocabulary_size:
        raise ValueError("vocabulary size disagrees with metadata")
    originals = set()
    for row in vocabulary:
        original = _integer(row["original_id"], "original_id", maximum=2**31-1)
        if original in originals:
            raise ValueError("duplicate original vocabulary ID")
        originals.add(original)
        if not isinstance(row["hex"], str) or re.fullmatch(r"(?:[0-9a-f]{2})+", row["hex"]) is None:
            raise ValueError("vocabulary bytes require nonempty canonical lowercase hex")
    states = {}
    original_members = set()
    for row in model["states"]:
        state = _integer(row["id"], "state ID", minimum=vocabulary_size,
                         maximum=2**31-1)
        stage = _integer(row["stage"], "stage", maximum=2*layers)
        if state in states:
            raise ValueError("duplicate state ID")
        if len(row["bits"]) != width:
            raise ValueError("state width differs from model width")
        for bits in row["bits"]:
            _integer(bits, "state bits", maximum=65535)
        if "members" in row:
            members = row["members"]
            if (not isinstance(members, list) or not members or
                    row.get("member_count") != len(members)):
                raise ValueError("invalid original-state membership count")
            for member in members:
                _integer(member, "original state ID", minimum=vocabulary_size,
                         maximum=2**31-1)
                if member in original_members:
                    raise ValueError("original state belongs to multiple classes")
                original_members.add(member)
        states[state] = stage

    def state_at(state, stage):
        _integer(state, "referenced state", minimum=vocabulary_size,
                 maximum=2**31-1)
        if states.get(state) != stage:
            raise ValueError(f"state {state} does not belong to stage {stage}")

    entries = set()
    for token, position, state in model["entry"]:
        _integer(token, "entry token", maximum=vocabulary_size-1)
        _integer(position, "entry position", maximum=1023)
        if (token, position) in entries:
            raise ValueError("duplicate entry key")
        entries.add((token, position))
        state_at(state, 0)
    if not entries or len(model["attention"]) != layers or len(model["mlp"]) != layers:
        raise ValueError("missing entry table or wrong block count")
    for block in range(layers):
        keys = set()
        for prefix, output in model["attention"][block]:
            if not 1 <= len(prefix) <= 1024 or tuple(prefix) in keys:
                raise ValueError("invalid or duplicate attention history")
            keys.add(tuple(prefix))
            for state in prefix:
                state_at(state, 2*block)
            state_at(output, 2*block+1)
        keys = set()
        for key, output in model["mlp"][block]:
            if key in keys:
                raise ValueError("duplicate MLP key")
            keys.add(key)
            state_at(key, 2*block+1)
            state_at(output, 2*block+2)
    language_modeling_head_inputs = set()
    for key, output in model["language_modeling_head"]:
        if key in language_modeling_head_inputs:
            raise ValueError("duplicate language modeling head key")
        language_modeling_head_inputs.add(key)
        state_at(key, 2*layers)
        _integer(output, "language modeling head token", maximum=vocabulary_size-1)
    if not language_modeling_head_inputs or not model["samples"]:
        raise ValueError("missing readout or verification samples")
    for sample in model["samples"]:
        if not prompt <= len(sample["tokens"]) < 1024:
            raise ValueError("verification sample cannot fit prompt and EOS")
        for token in sample["tokens"]:
            _integer(token, "sample token", maximum=vocabulary_size-1)


def _literal(data):
    # Fixed-width octal escapes cannot consume a following hex digit. Printable
    # ASCII stays readable while arbitrary token bytes remain exactly intact.
    escaped = {9: r"\t", 10: r"\n", 13: r"\r", 34: r'\"', 92: r"\\"}
    return '"' + "".join(escaped.get(byte, chr(byte) if 32 <= byte <= 126
                                     else f"\\{byte:03o}")
                         for byte in data) + '"'


def _token_name(data, compact_id, eos_token):
    """A readable, legal C++ identifier with the compact ID as a unique suffix.

    Token bytes need not be UTF-8. Keep ASCII words recognizable, name common
    punctuation/whitespace, and spell every other byte in hex. The ID suffix
    disambiguates literal spellings, repeated bytes, and capped long names.
    These names are labels only: token IDs and byte encodings never change.
    """
    if compact_id == eos_token:
        return f"kEos_{compact_id}"
    punctuation = {
        9: "Tab", 10: "Newline", 13: "CarriageReturn", 32: "Space",
        33: "Exclamation", 34: "Quote", 35: "Hash", 36: "Dollar",
        37: "Percent", 38: "Ampersand", 39: "Apostrophe",
        40: "LeftParen", 41: "RightParen", 42: "Asterisk", 43: "Plus",
        44: "Comma", 45: "Hyphen", 46: "Period", 47: "Slash",
        58: "Colon", 59: "Semicolon", 60: "LessThan", 61: "Equals",
        62: "GreaterThan", 63: "Question", 64: "At",
        91: "LeftBracket", 92: "Backslash", 93: "RightBracket",
        94: "Caret", 95: "Underscore", 96: "Backtick",
        123: "LeftBrace", 124: "Pipe", 125: "RightBrace", 126: "Tilde",
    }
    pieces = []
    for piece in re.findall(rb"[A-Za-z0-9]+|[^A-Za-z0-9]", data):
        if piece.isalnum():
            pieces.append(piece.decode("ascii"))
        else:
            pieces.append(punctuation.get(piece[0], f"Byte{piece[0]:02X}"))
    stem = "_".join(pieces)[:80].rstrip("_")
    return f"k{stem}_{compact_id}"


def _source(body, header='"tables.h"', description="", vocabulary=False):
    comments = "".join(f"// {line}\n" for line in description.splitlines())
    includes = f"#include {header}\n"
    if vocabulary:
        includes += '#include "vocabulary_tokens.h"\n'
    return f"// Generated by discretize_emit.py; do not edit.\n{comments}{includes}\nnamespace {NAMESPACE} {{\n{body}\n}}  // namespace {NAMESPACE}\n"


def _array(typename, name, rows):
    rows = list(rows)
    # No zero-length arrays: a dummy is excluded from the logical row count.
    contents = []
    for row in rows or ["{}"]:
        if isinstance(row, tuple):
            value, comment = row
            contents.append(f"  {value},  // {comment}\n")
        else:
            contents.append(f"  {row},\n")
    return f"const {typename} {name}[] = {{\n" + "".join(contents) + "};\n"


def _span(name, count):
    return f"{{{name}, {count}}}"


def _transition_object(body, interface, factory):
    """Give a private pure lookup its boundary's public polymorphic interface.

    Each factory owns one stateless instance with static lifetime. Model stores
    references to these objects, so they outlive every lookup.
    """
    parameters, arguments = {
        "CausalAttention": ("absl::Span<const DiscreteHiddenState> history", "history"),
        "Map": ("DiscreteHiddenState state", "state"),
        "PositionEmbedding": ("DiscreteToken token, int32_t position", "token, position"),
    }[interface]
    implementation = factory.removeprefix("Generated") + "Impl"
    return f'''namespace {{
{body}
class {implementation} final : public {interface} {{
 public:
  std::optional<DiscreteHiddenState> operator()({parameters}) override {{
    return Lookup({arguments});
  }}
}};
}}  // namespace

{interface}& {factory}() {{
  static {implementation} instance;
  return instance;
}}
'''


def _state_source(name, rows, description, token_names=None):
    rows = sorted(rows)
    values = (f"{{{{{a}}}, static_cast<DiscreteHiddenState>(vocab::{token_names[b]})}}"
              if token_names is not None else f"{{{{{a}}}, {{{b}}}}}" for a, b in rows)
    body = "struct StateRow { DiscreteHiddenState input; DiscreteHiddenState output; };\n"
    body += _array("StateRow", "kRows", values)
    body += f'''// Binary search stays private; the runtime only invokes this pure lookup.
std::optional<DiscreteHiddenState> Lookup(DiscreteHiddenState state) {{
  size_t first = 0;
  size_t last = {len(rows)};
  while (first < last) {{
    const size_t middle = first + (last - first) / 2;
    if (kRows[middle].input < state)
      first = middle + 1;
    else
      last = middle;
  }}
  if (first == {len(rows)} || kRows[first].input != state)
    return {{}};
  return {{kRows[first].output}};
}}
'''
    return _source(_transition_object(body, "Map", name),
                   description=description + "\nRows: {input_state, output_state_or_token}, sorted by input_state.",
                   vocabulary=token_names is not None)


def _boundary_name(stage):
    if stage == 0:
        return "token_plus_position_embedding"
    block = (stage - 1) // 2
    return f"block_{block}." + ("after_attention_residual" if stage % 2
                               else "after_mlp_residual")


def _comment_label(data):
    # JSON escaping guarantees one physical comment line, even for control
    # bytes, quotes, backslashes, and comment-looking token contents.
    return json.dumps(data.decode("utf-8", errors="backslashreplace"), ensure_ascii=True)


def _render(model, include_state_index=False, compact_transitions=False):
    layers = model["layers"]
    vocab = model["vocabulary"]
    token_names = [_token_name(bytes.fromhex(row["hex"]), token, model["eos_token"])
                   for token, row in enumerate(vocab)]
    declarations = ["PositionEmbedding& GeneratedPositionEmbedding();", "absl::Span<const VocabularyRow> GeneratedVocabulary();", "Map& GeneratedLanguageModelingHead();"]
    for block in range(layers):
        declarations += [f"CausalAttention& GeneratedAttention{block}();", f"Map& GeneratedMlp{block}();"]
    files = {"tables.h": f'// Generated declarations.\n#pragma once\n#include "{RUNTIME}"\nnamespace {NAMESPACE} {{\n' + "\n".join(declarations) + f"\n}}  // namespace {NAMESPACE}\n"}
    tokens_header = (
        '// Generated by discretize_emit.py; do not edit.\n#pragma once\n'
        f'#include "{RUNTIME}"\n\nnamespace {NAMESPACE}::vocab {{\n'
        '// One constant per compact vocabulary token; values are not GPT-2 IDs.\n'
        '// Names retain token case, spell whitespace/punctuation, and end with\n'
        '// the compact ID to keep every spelling unique. Long stems are capped.\n'
        '// Comments show token bytes and original GPT-2 IDs; Eos names the readout\n'
        '// terminator. Internal residual-state IDs intentionally remain numeric.\n')
    for token, row in enumerate(vocab):
        label = _literal(bytes.fromhex(row["hex"]))
        tokens_header += (f'// {label}; original GPT-2 ID {row["original_id"]}.\n'
                          f'inline constexpr DiscreteToken {token_names[token]}{{{token}}};\n')
    files["vocabulary_tokens.h"] = tokens_header + f"}}  // namespace {NAMESPACE}::vocab\n"
    entry = sorted(model["entry"])
    body = "struct EntryRow { DiscreteToken token; int32_t position; DiscreteHiddenState state; };\n"
    body += _array("EntryRow", "kRows", (
        (f"{{vocab::{token_names[a]}, {b}, {{{c}}}}}", "token " + _comment_label(bytes.fromhex(vocab[a]["hex"])))
        for a, b, c in entry))
    body += f'''std::optional<DiscreteHiddenState> Lookup(DiscreteToken token, int32_t position) {{
  size_t first = 0;
  size_t last = {len(entry)};
  while (first < last) {{
    const size_t middle = first + (last - first) / 2;
    const auto& row = kRows[middle];
    if (row.token < token || (row.token == token && row.position < position))
      first = middle + 1;
    else
      last = middle;
  }}
  if (first == {len(entry)} || kRows[first].token != token ||
      kRows[first].position != position)
    return {{}};
  return {{kRows[first].state}};
}}'''
    files["entry.cc"] = _source(_transition_object(body, "PositionEmbedding", "GeneratedPositionEmbedding"),
        vocabulary=True,
        description="Entry boundary: compact token ID + absolute position -> token-plus-position state.\n"
                    "Rows: {compact_token, zero_based_position, output_state}; sorted by (token, position).\n"
                    "Token comments decode the input token only; they are not semantic state labels.")
    body = "namespace {\n" + _array("VocabularyRow", "kRows", (
        (f"{{{row['original_id']}, {{{_literal(bytes.fromhex(row['hex']))}, {len(row['hex'])//2}}}}}",
         f"vocab::{token_names[token]}") for token, row in enumerate(vocab))) + "}\n"
    files["vocabulary.cc"] = _source(body + f"absl::Span<const VocabularyRow> GeneratedVocabulary() {{ return {_span('kRows', len(vocab))}; }}",
        description="Vocabulary: array index is compact token ID; each row is {original_GPT2_ID, {exact_bytes, byte_count}}.\n"
                    "Readable ASCII and fixed-width octal escapes preserve every original token byte.")
    for block in range(layers):
        keys, rows = [], []
        for prefix, output in sorted(model["attention"][block]):
            rows.append(f"{{{len(keys)}, {len(prefix)}, {{{output}}}}}")
            keys.extend(prefix)
        body = "struct AttentionRow { uint32_t offset; uint32_t length; DiscreteHiddenState output; };\n"
        body += _array("DiscreteHiddenState", "kKeys", (f"{{{state}}}" for state in keys)) + _array("AttentionRow", "kRows", rows)
        body += f'''// Lexicographic comparison checks every symbol of the complete prefix.
int ComparePrefix(const AttentionRow& row, absl::Span<const DiscreteHiddenState> prefix) {{
  const size_t shared = row.length < prefix.size() ? row.length : prefix.size();
  for (size_t index = 0; index < shared; ++index) {{
    const DiscreteHiddenState state = kKeys[row.offset + index];
    if (state < prefix[index])
      return -1;
    if (state > prefix[index])
      return 1;
  }}
  return row.length < prefix.size() ? -1 : row.length > prefix.size() ? 1 : 0;
}}
std::optional<DiscreteHiddenState> Lookup(absl::Span<const DiscreteHiddenState> prefix) {{
  size_t first = 0;
  size_t last = {len(rows)};
  while (first < last) {{
    const size_t middle = first + (last - first) / 2;
    if (ComparePrefix(kRows[middle], prefix) < 0)
      first = middle + 1;
    else
      last = middle;
  }}
  if (first == {len(rows)} || ComparePrefix(kRows[first], prefix) != 0)
    return {{}};
  return {{kRows[first].output}};
}}
'''
        files[f"attention_{block}.cc"] = _source(_transition_object(body, "CausalAttention", f"GeneratedAttention{block}"),
            description=f"Block {block} attention boundary: complete causal state prefix at {_boundary_name(2*block)}\n"
                        f"-> current-position state at {_boundary_name(2*block+1)}.\n"
                        "kKeys concatenates exact full prefixes, including the current position.\n"
                        "Rows: {key_offset, prefix_length, output_state}; ordered lexicographically by full prefix.\n"
                        "Lookup compares every integer in the prefix.")
        files[f"mlp_{block}.cc"] = _state_source(f"GeneratedMlp{block}", model["mlp"][block],
            f"Block {block} pointwise MLP boundary: {_boundary_name(2*block+1)}\n"
            f"-> {_boundary_name(2*block+2)}. This preserves the MLP residual boundary.")
    files["language_modeling_head.cc"] = _state_source("GeneratedLanguageModelingHead", model["language_modeling_head"],
        f"Readout: final {_boundary_name(2*layers)} state -> compact next-token ID.\n"
        "Outputs are token IDs, including EOS; no continuation sequence is stored here.",
        token_names=token_names)
    body = "const Model& GeneratedModel() {\n"
    if layers:
        body += "  static " + _array("Transformer", "kTransformers", (
            f"{{GeneratedAttention{i}(), GeneratedMlp{i}()}}" for i in range(layers)))
    transformers = _span("kTransformers", layers) if layers else "{}"
    body += f"  static const Model model{{1024, {model['prompt_tokens']}, vocab::{token_names[model['eos_token']]}, GeneratedVocabulary(), {transformers}, GeneratedLanguageModelingHead(), GeneratedPositionEmbedding()}};\n  return model;\n}}"
    files["model.cc"] = _source(body, vocabulary=True, description=
        "Integer network in original boundary order: entry -> (attention residual -> MLP residual) per block -> language modeling head.\n"
        "All attention and MLP boundaries remain separate. State IDs never identify corpus lines.")
    files["prompt_encoder.cc"] = _render_encoder(model, token_names)
    files["verification.cc"] = _render_verification(model, token_names)
    named_tokens = ",\n".join(f"vocab::{name}" for name in token_names)
    files["generated_model_test.cc"] = f'''#include <sstream>
#include "{RUNTIME}"
#include "vocabulary_tokens.h"
#include "gtest/gtest.h"
namespace {NAMESPACE} {{
absl::Status VerifyGeneratedModel(const Model&, std::ostream&);
TEST(GeneratedIntegerModel, NamedVocabularyCoversEveryCompactId) {{
  constexpr DiscreteToken kTokens[] = {{
{named_tokens}
  }};
  ASSERT_EQ(absl::MakeConstSpan(kTokens).size(), GeneratedModel().vocabulary.size());
  for (size_t token = 0; token < absl::MakeConstSpan(kTokens).size(); ++token)
    EXPECT_EQ(kTokens[token].value, token);
}}
TEST(GeneratedIntegerModel, IndependentAutoregressiveCorpusVerification) {{
  const auto& model = GeneratedModel();
  ASSERT_TRUE(ValidateModel(model).ok());
  std::ostringstream report;
  auto result = VerifyGeneratedModel(model, report);
  EXPECT_TRUE(result.ok()) << result << "\\n" << report.str();
}}
}}  // namespace {NAMESPACE}
'''
    files["BUILD.bazel"] = _render_build(layers, compact_transitions)
    if include_state_index:
        files["state_index.tsv"] = _render_state_index(model)
        if all("members" in row for row in model["states"]):
            lines = ["# Inspection only: IDs refer to the exact, unreduced baseline.",
                     "state_id\tboundary\toriginal_state_ids_json"]
            for row in sorted(model["states"], key=lambda item: item["id"]):
                members = json.dumps(sorted(row["members"]), separators=(",", ":"))
                lines.append(f"{row['id']}\t{_boundary_name(row['stage'])}\t{members}")
            files["state_members.tsv"] = "\n".join(lines) + "\n"
    if compact_transitions:
        from .discretize_logic import render_compact
        render_compact(model, token_names, files, _source, _transition_object)
    manifest = {
        "schema": 1, "generator": "discretize_emit.py", "layers": layers,
        "states": len(model["states"]), "vocabulary_size": model["vocab_size"],
        "samples": len(model["samples"]), "scored_targets": sum(len(row["tokens"])-model["prompt_tokens"]+1 for row in model["samples"]),
        "integer_only_inference": True, "expected_tokens_target": "verification",
        "state_index_included": include_state_index,
        "transition_representation": "control_flow" if compact_transitions else "tables",
        "files": {name: hashlib.sha256(value.encode()).hexdigest() for name, value in sorted(files.items())},
    }
    files["manifest.json"] = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    return files, manifest


def _render_state_index(model):
    """Inspection only: replay complete samples once at every real position."""
    entries = {(token, position): state for token, position, state in model["entry"]}
    attention = [{tuple(prefix): output for prefix, output in table}
                 for table in model["attention"]]
    mlp = [dict(table) for table in model["mlp"]]
    counts, examples = {}, {}

    def observe(states, prefixes):
        for state, context in zip(states, prefixes):
            counts[state] = counts.get(state, 0) + 1
            selected = examples.setdefault(state, [])
            if len(selected) < 3 and context not in selected:
                selected.append(context)

    for sample in model["samples"]:
        tokens = sample["tokens"]
        prefixes = []
        text = b""
        for position, token in enumerate(tokens):
            text += bytes.fromhex(model["vocabulary"][token]["hex"])
            context = {"compact_ids": tokens[:position+1],
                       "text": text[:160].decode("utf-8", errors="backslashreplace")}
            if len(text) > 160:
                context["text_truncated_after_bytes"] = 160
            prefixes.append(context)
        try:
            states = [entries[token, position] for position, token in enumerate(tokens)]
            observe(states, prefixes)
            for block in range(model["layers"]):
                states = [attention[block][tuple(states[:position+1])]
                          for position in range(len(states))]
                observe(states, prefixes)
                states = [mlp[block][state] for state in states]
                observe(states, prefixes)
        except KeyError as error:
            raise ValueError(f"state-index replay encountered an unsupported key: {error}") from error
    lines = [
        "# Inspection only; never linked into or read by inference.",
        "# Examples are empirical captured token-prefix contexts, not semantic labels.",
        "# A representative vector is only one original member, not the meaning of a merged class; unknown membership counts are not guessed.",
        "# Occurrences count each real sample position once at its named boundary, including prompt positions; no padding or repeated autoregressive passes.",
        "# Each example has complete compact token IDs and text preview of at most 160 original bytes; at most three distinct examples per state.",
        "state_id\tboundary\trepresentative_bf16_hex\tobserved_occurrences\tempirical_prefix_examples_json\toriginal_member_count",
    ]
    for row in sorted(model["states"], key=lambda item: item["id"]):
        state = row["id"]
        vector = " ".join(f"{bits:04x}" for bits in row["bits"])
        contexts = json.dumps(examples.get(state, []), ensure_ascii=True, separators=(",", ":"))
        members = row.get("member_count", "unknown")
        lines.append(f"{state}\t{_boundary_name(row['stage'])}\t{vector}\t{counts.get(state, 0)}\t{contexts}\t{members}")
    return "\n".join(lines) + "\n"


def _render_encoder(model, token_names):
    prefixes = {}
    for sample in model["samples"]:
        text = b""
        for position, token in enumerate(sample["tokens"]):
            text += bytes.fromhex(model["vocabulary"][token]["hex"])
            ids = tuple(sample["tokens"][:position+1])
            if text in prefixes and prefixes[text] != ids:
                raise ValueError("ambiguous captured text-prefix encoding")
            prefixes[text] = ids
    tokens, rows = [], []
    for text, ids in sorted(prefixes.items()):
        rows.append(f"{{{{{_literal(text)}, {len(text)}}}, {len(tokens)}, {len(ids)}}}")
        tokens.extend(ids)
    body = "namespace {\nstruct PromptRow { absl::string_view text; size_t offset; size_t length; };\n"
    body += _array("DiscreteToken", "kTokens", (f"vocab::{token_names[token]}" for token in tokens)) + _array("PromptRow", "kPrompts", rows) + "}\n"
    body += f'''absl::StatusOr<std::vector<DiscreteToken>> EncodeGeneratedPrompt(absl::string_view text) {{
  auto row = std::lower_bound(std::begin(kPrompts), std::end(kPrompts), text,
      [](const PromptRow& r, absl::string_view key) {{ return r.text < key; }});
  if (row == std::end(kPrompts) || row->text != text)
    return absl::NotFoundError("unsupported text encoding: use a captured corpus prefix at a token boundary, or --token_ids");
  return std::vector<DiscreteToken>(kTokens + row->offset, kTokens + row->offset + row->length);
}}'''
    return '#include <algorithm>\n#include <iterator>\n' + _source(body, f'"{RUNTIME}"', vocabulary=True)


def _render_verification(model, token_names):
    tokens, rows = [], []
    for sample in model["samples"]:
        rows.append(f"{{{len(tokens)}, {len(sample['tokens'])}}}")
        tokens.extend(sample["tokens"])
    body = "namespace {\nstruct Sample { size_t offset; size_t length; };\n"
    body += _array("DiscreteToken", "kExpectedTokens", (f"vocab::{token_names[token]}" for token in tokens)) + _array("Sample", "kSamples", rows) + "}\n"
    body += '''absl::Status VerifyGeneratedModel(const Model& model, std::ostream& output) {
  size_t targets = 0;
  size_t sentences = 0;
  for (const auto& sample : kSamples) {
    const auto original = absl::MakeConstSpan(kExpectedTokens + sample.offset, sample.length);
    auto generated = Generate(model, original.first(model.prompt_tokens),
                              sample.length - model.prompt_tokens + 1);
    if (!generated.ok()) return generated.status();
    std::vector<DiscreteToken> expected(original.begin() + model.prompt_tokens, original.end());
    expected.push_back(model.eos_token);
    if (*generated != expected) {
      output << "mismatch at verification sentence " << sentences + 1 << "\\n";
      return absl::DataLossError("integer autoregressive completion differs from independent fixture");
    }
    targets += expected.size();
    ++sentences;
  }
  output << "errors=0/" << targets << " exact_sentences=" << sentences << "/" << sentences
         << " eos=" << sentences << " autoregressive=true integer_only=true\\n";
  return absl::OkStatus();
}'''
    return '#include <ostream>\n' + _source(body, f'"{RUNTIME}"', vocabulary=True)


def _render_build(layers, compact_transitions=False):
    sources = ["entry.cc", "model.cc", "language_modeling_head.cc", "vocabulary.cc"]
    sources += [f"attention_{i}.cc" for i in range(layers)]
    sources += [f"mlp_{i}.cc" for i in range(layers)]
    source_text = "\n".join(f'        "{name}",' for name in sorted(sources))
    text = f'''# Generated ordinary C++ targets; no build-time generation or GPU dependency.
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

cc_library(
    name = "vocabulary_tokens",
    hdrs = ["vocabulary_tokens.h"],
    visibility = ["//visibility:public"],
    deps = ["{PACKAGE}:runtime"],
)
cc_library(
    name = "model",
    srcs = [
{source_text}
    ],
    hdrs = ["tables.h"],
    visibility = ["//visibility:public"],
    deps = [":vocabulary_tokens", "{PACKAGE}:runtime"],
)
cc_library(
    name = "prompt_encoder",
    srcs = ["prompt_encoder.cc"],
    deps = [":vocabulary_tokens", "{PACKAGE}:runtime"],
)
cc_library(
    name = "verification",
    srcs = ["verification.cc"],
    deps = [":vocabulary_tokens", "{PACKAGE}:runtime"],
)
cc_binary(
    name = "discretized_model",
    srcs = ["{PACKAGE}:main.cc"],
    deps = [":model", ":prompt_encoder", ":verification", "{PACKAGE}:runtime"],
)
cc_test(
    name = "generated_model_test",
    srcs = ["generated_model_test.cc"],
    deps = [":model", ":verification", ":vocabulary_tokens", "{PACKAGE}:runtime", "@googletest//:gtest_main"],
)
'''
    if compact_transitions:
        text += f'''
# Independent per-boundary fixtures are test-only, never linked by the CLI.
cc_test(
    name = "generated_transition_test",
    srcs = ["generated_transition_test.cc"],
    deps = [":model", ":vocabulary_tokens", "{PACKAGE}:runtime", "@googletest//:gtest_main"],
)
'''
    return text


def _publish(staging, destination):
    rename = ctypes.CDLL(None, use_errno=True).renameat2
    rename.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(staging), -100, os.fsencode(destination), 1):
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(destination))


def emit_model(model, output_dir, *, include_state_index=False,
               compact_transitions=False):
    """Emit into a fresh directory; optional state index is inspection-only."""
    _validate(model)
    if type(include_state_index) is not bool:
        raise ValueError("include_state_index must be a boolean")
    if type(compact_transitions) is not bool:
        raise ValueError("compact_transitions must be a boolean")
    files, manifest = _render(model, include_state_index=include_state_index,
                              compact_transitions=compact_transitions)
    destination = Path(output_dir)
    if not destination.name or destination.name in (".", ".."):
        raise ValueError("output must name a fresh directory")
    destination = destination.parent.resolve(strict=True) / destination.name
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing existing generated output: {destination}")
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}.emit-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        for name, contents in sorted(files.items()):
            with (staging / name).open("x", encoding="utf-8", newline="\n") as output:
                output.write(contents)
                output.flush()
                os.fsync(output.fileno())
        _publish(staging, destination)
    return manifest
