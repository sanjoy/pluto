"""Replace individual generated tables with exact-domain transition programs.

This is a representation pass, not another state merger. Each attention, MLP,
entry and readout function retains its boundary. Test fixtures are emitted into
a separate target and never become inputs to the production functions.
"""

import json

from .discretize_attention_logic import render_attention
from .discretize_pointwise import render_entry, render_pointwise
from .discretize_transition_tests import render_transition_test


def render_compact(model, token_names, files, source, transition_object):
    """Rewrite files using the emitter's source and interface-object wrappers."""
    names = ["vocab::" + name for name in token_names]
    reports = []

    def install(filename, body, stats, description, vocabulary=False):
        before = len(files[filename].encode())
        files[filename] = source(body, description=description,
                                 vocabulary=vocabulary)
        reports.append({"file": filename, **stats,
                        "unformatted_source_bytes_before": before,
                        "unformatted_source_bytes_after": len(files[filename].encode())})

    body, stats = render_entry("Lookup", model["entry"], names)
    body = transition_object(body, "PositionEmbedding", "GeneratedPositionEmbedding")
    install("entry.cc", body, stats,
            "Entry: compact token and absolute position -> residual symbol.\n"
            "A default symbol plus exceptional positions describes each token;\n"
            "support masks reject every token/position absent from the source.", True)
    for block, rows in enumerate(model["attention"]):
        body, stats = render_attention("Lookup", rows)
        body = transition_object(body, "CausalAttention", f"GeneratedAttention{block}")
        install(f"attention_{block}.cc", body, stats,
                f"Block {block}: exact causal-history decision program.\n"
                "Shared suffixes and sequence checks compress this boundary only.\n"
                "No neighboring layer, sentence identity or future token is consulted.")
        body, stats = render_pointwise("Lookup", model["mlp"][block])
        body = transition_object(body, "Map", f"GeneratedMlp{block}")
        install(f"mlp_{block}.cc", body, stats,
                f"Block {block}: pointwise MLP residual transition.\n"
                "Symbol renaming may expose a guarded offset. This function and\n"
                "both boundary alphabets remain separate; it is not a layer bypass.")
    body, stats = render_pointwise("Lookup", model["language_modeling_head"], names)
    body = transition_object(body, "Map", "GeneratedLanguageModelingHead")
    install("language_modeling_head.cc", body, stats,
            "Final residual symbol -> named vocabulary token.\n"
            "No unobserved input is assigned a default prediction.", True)
    files["transition_patterns.json"] = json.dumps({
        "schema": 1,
        "scope": "exact individual transition domains; no cross-layer folding",
        "source_measurement": "unformatted bytes; not executable size",
        "transitions": reports,
        "relabeling": model.get("stats", {}).get("pointwise_relabeling", {}),
    }, indent=2, sort_keys=True) + "\n"
    if "state_relabeling" in model:
        lines = ["# Pure within-boundary renaming relative to generator input.",
                 "old_state_id\tnew_state_id\tboundary_index"]
        lines += ["\t".join(map(str, row)) for row in model["state_relabeling"]]
        files["state_relabeling.tsv"] = "\n".join(lines) + "\n"
    files["generated_transition_test.cc"] = render_transition_test(model, names)
