#!/usr/bin/env python3
"""Compile finite attention maps into exact, readable acyclic control flow.

The recognizer consumes the complete ordered history. Missing transitions and
nonaccepting prefixes are unsupported, including extensions of accepted leaves.
Equivalent suffix programs share code only when both acceptance/output and all
future labeled transitions agree. Recognizer nodes are implementation locations,
not new residual-state classes or replacements for transformer boundaries.
"""

from bisect import bisect_left
from dataclasses import dataclass
import re


_CPP_KEYWORDS = frozenset("""
alignas alignof and and_eq asm auto bitand bitor bool break case catch char
char8_t char16_t char32_t class compl concept const consteval constexpr constinit
const_cast continue co_await co_return co_yield decltype default delete do double
dynamic_cast else enum explicit export extern false float for friend goto if
inline int long mutable namespace new noexcept not not_eq nullptr operator or
or_eq private protected public register reinterpret_cast requires return short
signed sizeof static static_assert static_cast struct switch template this
thread_local throw true try typedef typeid typename union unsigned using virtual
void volatile wchar_t while xor xor_eq
""".split())


@dataclass(frozen=True)
class AttentionNode:
    output: int | None
    edges: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class AttentionProgram:
    nodes: tuple[AttentionNode, ...]
    root: int
    rows: int
    key_scalars: int
    trie_nodes: int


def _state(value):
    if type(value) is not int or not 0 <= value <= 0x7fffffff:
        raise ValueError("attention symbols and outputs must be nonnegative int32 integers")
    return value


def build_attention(rows):
    """Build a deterministic minimal acyclic output recognizer from key/value rows.

    Duplicate identical rows are harmless; conflicting keys are errors. Keys
    must be nonempty. The empty table compiles to an always-unsupported program.
    Inputs are never mutated, and every component of the result is immutable.
    """
    table = {}
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            raise ValueError("attention row must contain a history and output")
        history, output = row
        if not isinstance(history, (list, tuple)) or not history:
            raise ValueError("attention history must be a nonempty sequence")
        key = tuple(_state(symbol) for symbol in history)
        _state(output)
        if key in table and table[key] != output:
            raise ValueError("conflicting attention outputs for the same history")
        table[key] = output

    children = [{}]
    outputs = [None]
    for key, output in sorted(table.items()):
        node = 0
        for symbol in key:
            if symbol not in children[node]:
                children[node][symbol] = len(children)
                children.append({})
                outputs.append(None)
            node = children[node][symbol]
        outputs[node] = output

    canonical = {}
    nodes = []
    mapped = [0] * len(children)
    # Parents are allocated before children. Bottom-up structural interning
    # preserves the complete accepted language and its output at every key.
    for node in reversed(range(len(children))):
        signature = AttentionNode(outputs[node], tuple(sorted(
            (symbol, mapped[child]) for symbol, child in children[node].items())))
        if signature not in canonical:
            canonical[signature] = len(nodes)
            nodes.append(signature)
        mapped[node] = canonical[signature]
    return AttentionProgram(tuple(nodes), mapped[0], len(table),
                            sum(map(len, table)), len(children))


def evaluate(program, prefix):
    """Return the exact output state, or None for any unsupported history."""
    node = program.root
    for symbol in prefix:
        if type(symbol) is not int or not 0 <= symbol <= 0x7fffffff:
            return None
        edges = program.nodes[node].edges
        index = bisect_left(edges, (symbol, -1))
        if index == len(edges) or edges[index][0] != symbol:
            return None
        node = edges[index][1]
    return program.nodes[node].output


def _layout(program, chunk_size):
    parents = [[] for _ in program.nodes]
    for source, node in enumerate(program.nodes):
        for _, target in node.edges:
            parents[target].append(source)
    starts = set()
    entries = {}
    for node, incoming in enumerate(parents):
        chunk = node // chunk_size
        entry = node == program.root or any(parent // chunk_size != chunk for parent in incoming)
        if entry:
            entries.setdefault(chunk, []).append(node)
        if (entry or len(incoming) != 1 or
                len(program.nodes[incoming[0]].edges) != 1):
            starts.add(node)
    return starts, entries


def render_attention(name, rows, *, chunk_size=256, strategy="hybrid"):
    """Return (C++ definitions, statistics) for an exact attention callback.

    The public signature is:
      TransitionResult name(absl::Span<const DiscreteHiddenState> history)
    TransitionResult contains std::optional<DiscreteHiddenState> output: a value (including
    zero) is a supported output, and std::nullopt is an unsupported history. The
    caller supplies its declaration and standard size/integer types via runtime.h.

    Generated MATCH statements are literal symbol tests, not packed old-table
    rows. Hybrid emission factors longer unary runs into short typed sequences;
    control_flow emission uses only literal tests. Shared suffixes use local
    labels; branching nodes use switches. Helpers
    have bounded node counts, with explicit continuation dispatch between them,
    so a whole corpus does not become one enormous optimizer control-flow graph.
    """
    if (not isinstance(name, str) or
            re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) is None or "__" in name or
            name in _CPP_KEYWORDS):
        raise ValueError("attention function name must be a nonreserved C++ identifier")
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if strategy not in ("hybrid", "control_flow"):
        raise ValueError("strategy must be hybrid or control_flow")
    program = build_attention(rows)
    starts, entries = _layout(program, chunk_size)
    result_type = name + "Step"
    done = "k" + name + "Done"
    lines = [
        "// Exact ordered-history recognizer; unknown paths are rejected.",
        "// Control labels share identical suffix programs, not neural states.",
        "// MATCH checks one symbol; ending before it returns this prefix's output.",
        "// No hash, nearest-state fallback, corpus ID, or answer cache is used.",
        "namespace {",
        f"constexpr std::uint32_t {done} = 0xffffffffu;",
        f"struct {result_type} {{ std::uint32_t next; TransitionResult result; }};",
        "#define PLUTO_ATTN_END(output) \\",
        f"  do {{ if (position == history.size()) return {{{done}, {{DiscreteHiddenState{{output}}}}}}; }} while (false)",
        "#define PLUTO_ATTN_MORE() \\",
        f"  do {{ if (position == history.size()) return {{{done}, {{}}}}; }} while (false)",
        "#define PLUTO_ATTN_MATCH(symbol, output) \\",
        f"  do {{ PLUTO_ATTN_END(output); if (history[position++].value != symbol) return {{{done}, {{}}}}; }} while (false)",
        "#define PLUTO_ATTN_SKIP(symbol) \\",
        f"  do {{ PLUTO_ATTN_MORE(); if (history[position++].value != symbol) return {{{done}, {{}}}}; }} while (false)",
    ]

    def jump(target, chunk):
        return (f"goto n{target};" if target // chunk_size == chunk else
                f"return {{{target}u, {{}}}};")

    sequence_insert = len(lines)
    sequences = {}
    sequence_calls = 0
    visited = set()
    for chunk, entry_nodes in sorted(entries.items()):
        lines += [f"{result_type} {name}Part{chunk}(std::uint32_t node,",
                  "    [[maybe_unused]] absl::Span<const DiscreteHiddenState> history,",
                  "    [[maybe_unused]] std::size_t& position) {",
                  "  switch (node) {"]
        lines += [f"    case {node}u: goto n{node};" for node in sorted(entry_nodes, reverse=True)]
        lines += [f"    default: return {{{done}, {{}}}};", "  }"]
        for first in sorted((node for node in starts if node // chunk_size == chunk), reverse=True):
            lines.append(f"n{first}:")
            node_id = first
            while True:
                run = []
                run_nodes = []
                target = node_id
                if strategy == "hybrid":
                    while True:
                        candidate = program.nodes[target]
                        if candidate.output is None or len(candidate.edges) != 1:
                            break
                        symbol, following = candidate.edges[0]
                        run.append((symbol, candidate.output))
                        run_nodes.append(target)
                        target = following
                        if target in starts or target // chunk_size != chunk:
                            break
                if len(run) >= 4:
                    pattern = tuple(run)
                    if pattern not in sequences:
                        sequences[pattern] = len(sequences)
                    sequence_calls += 1
                    for consumed in run_nodes:
                        if consumed in visited:
                            raise AssertionError("attention sequence emitted twice")
                        visited.add(consumed)
                    lines.append(f"  PLUTO_ATTN_RUN(k{name}Sequence{sequences[pattern]});")
                    if target not in starts and target // chunk_size == chunk:
                        node_id = target
                        continue
                    lines.append("  " + jump(target, chunk))
                    break
                if node_id in visited:
                    raise AssertionError("attention control-flow block emitted twice")
                visited.add(node_id)
                node = program.nodes[node_id]
                if not node.edges:
                    if node.output is not None:
                        lines.append(f"  PLUTO_ATTN_END({node.output});")
                    lines.append(f"  return {{{done}, {{}}}};")
                    break
                if len(node.edges) == 1:
                    symbol, target = node.edges[0]
                    lines.append(f"  PLUTO_ATTN_SKIP({symbol});" if node.output is None else
                                 f"  PLUTO_ATTN_MATCH({symbol}, {node.output});")
                    if target not in starts and target // chunk_size == chunk:
                        node_id = target
                        continue
                    lines.append("  " + jump(target, chunk))
                    break
                lines.append("  PLUTO_ATTN_MORE();" if node.output is None else
                             f"  PLUTO_ATTN_END({node.output});")
                lines.append("  switch (history[position++].value) {")
                targets = {}
                for symbol, target in node.edges:
                    targets.setdefault(target, []).append(symbol)
                for target, symbols in targets.items():
                    lines.append("    " + " ".join(f"case {symbol}:" for symbol in symbols))
                    lines.append("      " + jump(target, chunk))
                lines += [f"    default: return {{{done}, {{}}}};", "  }"]
                break
        lines.append("}")
    if len(visited) != len(program.nodes):
        raise AssertionError("attention control-flow layout omitted a reachable node")
    sequence_bits = 16 if all(value <= 65535 for pattern in sequences
                              for pair in pattern for value in pair) else 32
    if sequences:
        # Keeping this matcher out of line avoids re-expanding literal runs
        # into thousands of duplicate machine-code tests at optimization time.
        # The standard attribute syntax is recognized by GCC and Clang; other
        # conforming compilers may ignore the vendor attribute without changing
        # the program's semantics.
        sequence_type = name + "MatchStep"
        sequence_lines = [
            f"struct {sequence_type} {{ std::uint{sequence_bits}_t symbol, output; }};",
            "[[gnu::noinline]]",
            f"{result_type} {name}MatchSequence(const {sequence_type}* steps,",
            "    std::size_t count, absl::Span<const DiscreteHiddenState> history, std::size_t& position) {",
            "  for (std::size_t index = 0; index < count; ++index) {",
            f"    if (position == history.size()) return {{{done}, {{DiscreteHiddenState{{static_cast<int>(steps[index].output)}}}}}};",
            f"    if (history[position++].value != static_cast<int>(steps[index].symbol)) return {{{done}, {{}}}};",
            "  }",
            "  return {0u, {}};  // The literal run matched; continue at its shared tail.",
            "}",
            "#define PLUTO_ATTN_RUN(steps) \\",
            f"  do {{ auto matched = {name}MatchSequence(steps, sizeof(steps) / sizeof(steps[0]), history, position); if (matched.next == {done}) return matched; }} while (false)",
        ]
        for pattern, index in sequences.items():
            sequence_lines.append(f"constexpr {sequence_type} k{name}Sequence{index}[] = {{")
            sequence_lines += [f"  {{{symbol}u, {output}u}}," for symbol, output in pattern]
            sequence_lines.append("};")
        lines[sequence_insert:sequence_insert] = sequence_lines
    lines += ["#undef PLUTO_ATTN_END", "#undef PLUTO_ATTN_MORE",
              "#undef PLUTO_ATTN_MATCH", "#undef PLUTO_ATTN_SKIP", "#undef PLUTO_ATTN_RUN",
              "}  // namespace", "",
              f"TransitionResult {name}(absl::Span<const DiscreteHiddenState> history) {{",
              "  std::size_t position = 0;",
              f"  std::uint32_t node = {program.root}u;",
              "  for (;;) {",
              f"    {result_type} step;",
              f"    switch (node / {chunk_size}u) {{"]
    for chunk in sorted(entries):
        lines.append(f"      case {chunk}u: step = {name}Part{chunk}(node, history, position); break;")
    lines += ["      default: return {};", "    }",
              f"    if (step.next == {done}) return step.result;",
              "    node = step.next;", "  }", "}"]
    source = "\n".join(lines) + "\n"
    edges = sum(len(node.edges) for node in program.nodes)
    branch_edges = sum(len(node.edges) for node in program.nodes if len(node.edges) > 1)
    stats = {
        "representation": "shared_suffix_control_flow",
        "strategy": strategy,
        "rows": program.rows,
        "flat_key_scalars": program.key_scalars,
        "flat_scalars": program.key_scalars + 3 * program.rows,
        "trie_nodes": program.trie_nodes,
        "nodes": len(program.nodes),
        "edges": edges,
        "unary_nodes": sum(len(node.edges) == 1 for node in program.nodes),
        "branch_nodes": sum(len(node.edges) > 1 for node in program.nodes),
        "branch_edges": branch_edges,
        "control_blocks": len(starts),
        "helpers": len(entries),
        "helper_node_limit": chunk_size,
        "entry_cases": sum(map(len, entries.values())),
        "scalar_estimate": 3 * len(program.nodes) + 2 * branch_edges,
        "literal_sequence_patterns": len(sequences),
        "literal_sequence_steps": sum(map(len, sequences)),
        "literal_sequence_calls": sequence_calls,
        "literal_sequence_word_bits": sequence_bits,
        "source_bytes": len(source.encode("utf-8")),
    }
    return source, stats
