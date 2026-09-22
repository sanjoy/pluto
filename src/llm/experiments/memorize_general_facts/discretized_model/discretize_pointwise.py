#!/usr/bin/env python3
"""Exact within-boundary renaming and compact pointwise C++ generators.

Integer state labels carry no arithmetic meaning. Renaming an MLP's output
alphabet can expose its bijection as an offset without eliminating that MLP or
combining adjacent layers. Entry support masks remain part of the function.
"""

from collections import Counter, defaultdict
import copy


def _mapping(rows):
    result = {}
    for source, output in rows:
        if source in result and result[source] != output:
            raise ValueError("conflicting pointwise mapping")
        result[source] = output
    return result


def evaluate_pointwise(rows, state):
    """Return the exact (output, supported) result, with 0 for unsupported."""
    value = _mapping(rows).get(state)
    return (0, False) if value is None else (value, True)


def evaluate_entry(rows, token, position):
    return evaluate_pointwise([((t, p), out) for t, p, out in rows], (token, position))


def relabel_mlp_outputs(model):
    """Deep-copy and rename bijective MLP and terminal output alphabets.

    Mapping rows cover every state, including identities: [old_id,new_id,stage].
    Original member IDs, vectors, statistics and boundary ownership survive.
    No vocabulary token or original membership ID is renamed. When the complete
    final MLP and snap are bijections, their last two alphabets use disjoint
    vocabulary-sized ID ranges. These labels encode token IDs; their arithmetic
    is a presentation choice, not a discovery of neural-head linearity.
    """
    result = copy.deepcopy(model)
    states = {row["id"]: row for row in model["states"]}
    renaming = {state: state for state in states}
    eligible, skipped = [], []
    for layer, rows in enumerate(model["mlp"]):
        mapping = _mapping(rows)
        inputs, outputs = sorted(mapping), sorted(set(mapping.values()))
        if (not inputs or len(inputs) != len(outputs)
                or len(inputs) != inputs[-1] - inputs[0] + 1
                or len(outputs) != outputs[-1] - outputs[0] + 1
                or set(inputs) != {state for state, row in states.items() if row["stage"] == 2 * layer + 1}
                or set(outputs) != {state for state, row in states.items() if row["stage"] == 2 * layer + 2}):
            skipped.append(layer)
            continue
        eligible.append(layer)
        for source, output in mapping.items():
            renaming[output] = outputs[0] + source - inputs[0]
    alignment = {"vocabulary_aligned_final_boundaries": False}
    if model["layers"] > 0:
        last_stage = 2 * model["layers"]
        before = {state for state, row in states.items() if row["stage"] == last_stage - 1}
        after = {state for state, row in states.items() if row["stage"] == last_stage}
        last_mlp = _mapping(model["mlp"][-1])
        snap = _mapping(model["snap"])
        labels = set(snap.values())
        complete = (before and set(last_mlp) == before and set(last_mlp.values()) == after
                    and len(before) == len(after) and set(snap) == after
                    and len(labels) == len(after)
                    and all(type(token) is int and 0 <= token < model["vocab_size"] for token in labels))
        prefix_max = max([model["vocab_size"] - 1] +
                         [state for state, row in states.items() if row["stage"] < last_stage - 1])
        attention_base = prefix_max + 1
        final_base = attention_base + model["vocab_size"]
        if complete and final_base + model["vocab_size"] - 1 <= 0xffffffff:
            for state, output in last_mlp.items():
                renaming[state] = attention_base + snap[output]
            for state, token in snap.items():
                renaming[state] = final_base + token
            alignment = {"vocabulary_aligned_final_boundaries": True,
                         "last_attention_base": attention_base,
                         "final_state_base": final_base,
                         "reserved_range_size": model["vocab_size"]}
    for row in result["states"]:
        row["id"] = renaming[row["id"]]
    result["states"].sort(key=lambda row: (row["stage"], row["id"]))
    result["entry"] = sorted([[token, position, renaming[out]] for token, position, out in model["entry"]])
    result["attention"] = [sorted([[ [renaming[state] for state in prefix], renaming[out]]
                                     for prefix, out in table]) for table in model["attention"]]
    result["mlp"] = [sorted([[renaming[source], renaming[out]] for source, out in table])
                     for table in model["mlp"]]
    result["snap"] = sorted([[renaming[state], token] for state, token in model["snap"]])
    result.setdefault("stats", {})["pointwise_relabeling"] = {
        "eligible_layers": eligible, "skipped_layers": skipped,
        "changed_states": sum(old != new for old, new in renaming.items()),
        "layer_boundaries_preserved": True, **alignment}
    mapping_rows = [[old, renaming[old], states[old]["stage"]] for old in sorted(states)]
    return result, mapping_rows


def _name(value, token_names):
    return str(value) if token_names is None else token_names[value]


def _array(name, ctype, values, *, columns=12):
    rows = ["  static constexpr " + ctype + " " + name + "[] = {"]
    for offset in range(0, len(values), columns):
        rows.append("    " + ", ".join(str(value) for value in values[offset:offset + columns]) + ",")
    rows.append("  };")
    return rows


def _finish(lines, **stats):
    body = "\n".join(lines) + "\n"
    stats["source_bytes"] = len(body.encode("utf-8"))
    return body, stats


def _state_guard(low, high):
    conditions = []
    if low > 0:
        conditions.append(f"state < {low}u")
    if high < 0xffffffff:
        conditions.append(f"state > {high}u")
    return [] if not conditions else ["  if (" + " || ".join(conditions) + ") return {};"]


def _affine_expression(source, output, token_names):
    if token_names is not None:
        return f"static_cast<StateId>({_name(output, token_names)}) + (state - {source}u)"
    delta = output - source
    return "state" if delta == 0 else f"state {'+' if delta > 0 else '-'} {abs(delta)}u"


def render_pointwise(name, rows, token_names=None):
    """Return (complete C++ function, honest representation/size statistics).

    Numeric outputs use exact affine runs when short. Irregular dense domains
    use a guarded uint16 output vector when possible. Named vocabulary outputs
    retain vocab:: expressions. Sparse affine named maps use a named anchor and
    exact support mask; this arithmetic describes encoded labels, not weights.
    """
    mapping = _mapping(rows)
    ordered = sorted(mapping.items())
    if any(type(source) is not int or type(output) is not int
           or not 0 <= source <= 0xffffffff or not 0 <= output <= 0xffffffff
           for source, output in ordered):
        raise ValueError("pointwise IDs must fit uint32_t")
    begin = [f"TransitionResult {name}(StateId state) {{"]
    if not ordered:
        return _finish(begin + ["  (void)state;", "  return {};", "}"],
                       representation="empty", rows=0, table_bytes=0)
    low, high = ordered[0][0], ordered[-1][0]
    guard = _state_guard(low, high)
    affine = all(output - source == ordered[0][1] - low for source, output in ordered)
    dense = len(ordered) == high - low + 1
    if affine and dense:
        lines = begin + guard
        if token_names is not None:
            lines.append("  // State labels encode vocabulary IDs; no neural-head linearity is implied.")
        lines += [f"  return {_affine_expression(low, ordered[0][1], token_names)};", "}"]
        return _finish(lines, representation="guarded_affine", rows=len(ordered), affine_ranges=1,
                       table_bytes=0, named_anchor=token_names is not None)
    affine_candidate = None
    support_bytes = (high - low + 8) // 8
    if affine and support_bytes <= min(1_048_576, len(ordered) * 8):
        support = [0] * support_bytes
        for state, _ in ordered:
            index = state - low
            support[index // 8] |= 1 << (index % 8)
        lines = begin + guard
        if token_names is not None:
            lines.append("  // State labels encode vocabulary IDs; no neural-head linearity is implied.")
        lines += _array("kSupport", "uint8_t", [f"0x{value:02x}u" for value in support], columns=16)
        lines += [f"  const uint32_t offset = state - {low}u;",
                  "  if ((kSupport[offset >> 3] & (uint32_t{1} << (offset & 7u))) == 0) return {};",
                  f"  return {_affine_expression(low, ordered[0][1], token_names)};", "}"]
        affine_candidate = _finish(lines, representation="sparse_affine_support_mask", rows=len(ordered),
            table_bytes=support_bytes, supported_span=high - low + 1, named_anchor=token_names is not None)
    runs = []
    for source, output in ordered:
        delta = output - source
        if (token_names is None and runs and source == runs[-1][1] + 1 and delta == runs[-1][2]):
            runs[-1][1] = source
        else:
            runs.append([source, source, delta])
    # The explicit branch candidate leaves unsupported holes unsupported.
    branches = begin + guard
    singletons = []
    for first, last, delta in runs:
        if last - first >= 2:
            expression = _affine_expression(first, first + delta, None)
            conditions = ([] if first == 0 else [f"state >= {first}u"])
            if last < 0xffffffff:
                conditions.append(f"state <= {last}u")
            condition = " && ".join(conditions) if conditions else "true"
            branches.append(f"  if ({condition}) return {expression};")
        else:
            singletons.extend((state, mapping[state]) for state in range(first, last + 1))
    if singletons:
        branches.append("  switch (state) {")
        for source, output in singletons:
            branches.append(f"    case {source}: return {_name(output, token_names)};")
        branches += ["    default: return {};", "  }"]
    else:
        branches.append("  return {};")
    branches.append("}")
    branch_body, branch_stats = _finish(branches, representation="affine_ranges_and_switch",
        rows=len(ordered), affine_ranges=sum(last - first >= 2 for first, last, _ in runs),
        switch_cases=len(singletons), table_bytes=0)
    if affine_candidate is not None and len(affine_candidate[0]) < len(branch_body):
        return affine_candidate
    if dense:
        small = all(0 <= output <= 65535 for _, output in ordered)
        ctype = "uint16_t" if small else "StateId"
        vector = begin + guard + _array("kOutputs", ctype,
            [_name(output, token_names) for _, output in ordered])
        vector += [f"  return kOutputs[state - {low}];", "}"]
        vector_body, vector_stats = _finish(vector, representation="guarded_output_array",
            rows=len(ordered), table_bytes=len(ordered) * (2 if small else 4), named_outputs=token_names is not None)
        if len(vector_body) < len(branch_body):
            return vector_body, vector_stats
    return branch_body, branch_stats


def render_entry(name, rows, token_names):
    """Factor exact entry support masks, token defaults and sparse exceptions.

    A dictionary deduplicates packed (position mask, default state) patterns.
    Large or sparse token domains and positions beyond 63 use exact switches.
    """
    entry = {}
    for token, position, output in rows:
        if token < 0 or position < 0:
            raise ValueError("entry token and position must be nonnegative")
        key = (token, position)
        if key in entry and entry[key] != output:
            raise ValueError("conflicting entry mapping")
        entry[key] = output
    begin = [f"TransitionResult {name}(TokenId token, uint32_t position) {{"]
    if not entry:
        return _finish(begin + ["  (void)token;", "  (void)position;", "  return {};", "}"],
                       representation="empty", rows=0, table_bytes=0)
    tokens = defaultdict(dict)
    for (token, position), output in sorted(entry.items()):
        tokens[token][position] = output
    low, high = min(tokens), max(tokens)
    max_position = max(position for _, position in entry)
    states = sorted(set(entry.values()))
    state_index = {state: index for index, state in enumerate(states)}
    position_bits = max_position + 1
    state_bits = max(1, (len(states) - 1).bit_length())
    if position_bits + state_bits > 64 or high - low + 1 > max(64, 4 * len(tokens)):
        result = begin + ["  switch (token) {"]
        for token, positions in sorted(tokens.items()):
            result += [f"    case {_name(token, token_names)}:", "      switch (position) {"]
            result += [f"        case {position}: return {output};" for position, output in sorted(positions.items())]
            result += ["        default: return {};", "      }"]
        result += ["    default: return {};", "  }", "}"]
        return _finish(result, representation="exact_token_position_switch", rows=len(entry),
                       tokens=len(tokens), table_bytes=0)
    patterns = [0]
    pattern_index = {0: 0}
    indices, exceptions = [], []
    for token in range(low, high + 1):
        positions = tokens.get(token)
        if positions is None:
            indices.append(0)
            continue
        default = min(Counter(positions.values()).items(), key=lambda item: (-item[1], item[0]))[0]
        mask = sum(1 << position for position in positions)
        packed = (state_index[default] << position_bits) | mask
        if packed not in pattern_index:
            pattern_index[packed] = len(patterns)
            patterns.append(packed)
        indices.append(pattern_index[packed])
        exceptions.extend((token, position, output) for position, output in positions.items() if output != default)
    word_bytes = 4 if position_bits + state_bits <= 32 else 8
    word_type = "uint32_t" if word_bytes == 4 else "uint64_t"
    index_bytes = 1 if len(patterns) <= 256 else (2 if len(patterns) <= 65536 else 4)
    index_type = f"uint{index_bytes * 8}_t"
    suffix = "u" if word_bytes == 4 else "ull"
    lines = begin + [f"  if (token < {low} || token > {high} || position > {max_position}) return {{}};"]
    lines += _array("kTokenPatterns", index_type, indices)
    lines += _array("kPatterns", word_type, [f"0x{value:x}{suffix}" for value in patterns], columns=8)
    lines += [f"  const {word_type} packed = kPatterns[kTokenPatterns[token - {low}]];",
              f"  if ((packed & ({word_type}{{1}} << position)) == 0) return {{}};"]
    if exceptions:
        lines.append("  switch (token) {")
        grouped = defaultdict(list)
        for token, position, output in exceptions:
            grouped[token].append((position, output))
        for token, exceptional_positions in sorted(grouped.items()):
            lines.append(f"    case {_name(token, token_names)}:")
            for position, output in exceptional_positions:
                lines.append(f"      if (position == {position}) return {output};")
            lines.append("      break;")
        lines += ["    default: break;", "  }"]
    state_bytes = 0
    if len(states) == states[-1] - states[0] + 1:
        lines.append(f"  return {states[0]}u + static_cast<StateId>(packed >> {position_bits});")
    else:
        lines += _array("kStates", "StateId", states)
        state_bytes = 4 * len(states)
        lines.append(f"  return kStates[packed >> {position_bits}];")
    lines.append("}")
    return _finish(lines, representation="packed_support_patterns_and_exceptions", rows=len(entry),
        tokens=len(tokens), patterns=len(patterns), position_bits=position_bits,
        token_only_defaults=len(tokens), position_exceptions=len(exceptions),
        table_bytes=index_bytes * len(indices) + word_bytes * len(patterns) + state_bytes,
        named_exception_tokens=True)
