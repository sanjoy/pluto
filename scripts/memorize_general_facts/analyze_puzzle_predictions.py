#!/usr/bin/env python3
"""Audit the exact teacher-forced successes of puzzle_prediction_probe.

All counts concern the scored suffix plus EOS, not prompt/padding rows. Token
categories are deliberately mechanical: a fixed function-word list is a useful
contrast, not a semantic parser or a claim about whether a fact was learned.
The next-token baselines are fitted leave-one-row-out so rare singleton tokens
cannot obtain perfect accuracy by serving as their own training examples.
"""

import argparse
from collections import Counter, defaultdict
import csv
import html
import json
from pathlib import Path
import re
import statistics
import string


FUNCTION_WORDS = set("""
a an the and or but if while although because so than as of in on at by for
from with without into onto to through over under between among during before
after about around near across is are was were be been being am has have had
do does did can could will would shall should may might must not no nor both
either neither each every all some any many more most other another such that
this these those it its they their them he his him she her we our us you your
which who whom whose when where how what there here also only often usually
""".split())


def unescape(text):
    """Decode the probe's ASCII byte escapes, replacing split UTF-8 for display."""
    data = bytearray()
    index = 0
    escapes = {"\\": 92, "t": 9, "n": 10, "r": 13}
    while index < len(text):
        if text[index] != "\\":
            data.append(ord(text[index]))
            index += 1
            continue
        index += 1
        if index == len(text):
            raise ValueError("trailing backslash in TSV text")
        if text[index] == "x":
            value = text[index + 1:index + 3]
            if len(value) != 2 or any(c not in string.hexdigits for c in value):
                raise ValueError("invalid byte escape")
            data.append(int(value, 16))
            index += 3
        elif text[index] in escapes:
            data.append(escapes[text[index]])
            index += 1
        else:
            raise ValueError("unknown TSV escape")
    return data.decode("utf-8", errors="replace")


def category(text, prefix="", fact=None):
    word = text.strip()
    if word == "<|endoftext|>":
        return "EOS"
    if word and all(c in string.punctuation for c in word):
        return "Punctuation"
    if re.fullmatch(r"[0-9]+(?:[.,/-][0-9]+)*", word):
        return "Number pieces"
    # A BPE fragment such as `is` inside a name is not a function word.
    start = len(prefix) + len(text) - len(text.lstrip())
    end = len(prefix) + len(text.rstrip())
    whole_word = fact is None or (
        (start == 0 or not fact[start - 1].isalnum())
        and (end == len(fact) or not fact[end].isalnum()))
    if word.lower() in FUNCTION_WORDS and whole_word:
        return "Function-word tokens (fixed list)"
    return "Other word/subword tokens"


def load_rows(path):
    integers = ("fact_index", "position", "target", "prediction",
                "source_prediction", "fresh_prediction", "input_token",
                "target_rank")
    floats = ("target_probability", "top1_probability", "margin",
              "source_target_probability", "source_margin", "hidden_norm")
    with Path(path).open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t", quoting=csv.QUOTE_NONE))
    seen = set()
    for row in rows:
        for field in integers:
            row[field] = int(row[field])
        for field in floats:
            row[field] = float(row[field])
        for field in ("prefix", "target_text", "prediction_text"):
            row[field] = unescape(row[field])
        key = row["fact_index"], row["position"]
        if key in seen or row["position"] < 4 or row["target"] < 0:
            raise ValueError("duplicate or unscored prediction row")
        seen.add(key)
        row["correct"] = row["target"] == row["prediction"]
    # The row whose target is EOS has the complete original sentence as prefix.
    facts = {row["fact_index"]: row["prefix"] for row in rows
             if row["target_text"] == "<|endoftext|>"}
    if len(facts) != len({row["fact_index"] for row in rows}):
        raise ValueError("each fact must have an EOS target row")
    for row in rows:
        fact = facts[row["fact_index"]]
        if not fact.startswith(row["prefix"]):
            raise ValueError("prefix does not agree with fact text")
        if row["target_text"] != "<|endoftext|>" and not fact[len(row["prefix"]):].startswith(row["target_text"]):
            raise ValueError("target does not follow prefix")
        row["category"] = category(row["target_text"], row["prefix"], fact)
    return sorted(rows, key=lambda row: (row["fact_index"], row["position"]))


def metrics(rows):
    count = len(rows)
    correct = sum(row["correct"] for row in rows)
    return {"count": count, "correct": correct,
            "accuracy": correct / count if count else 0}


def group_metrics(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    return {name: metrics(group) for name, group in groups.items()}


def frequency_bin(count):
    if count == 1:
        return "1"
    if count == 2:
        return "2"
    for upper in (2, 5, 10, 20, 50, 100):
        if count <= upper:
            lower = {2: 2, 5: 3, 10: 6, 20: 11, 50: 21, 100: 51}[upper]
            return f"{lower}-{upper}"
    return "101+"


def leave_one_out_majority(rows, key):
    """Predict from peers, excluding this target; no peers means no prediction."""
    groups = defaultdict(Counter)
    for row in rows:
        groups[key(row)][row["target"]] += 1
    result = []
    for row in rows:
        counts = groups[key(row)].copy()
        counts[row["target"]] -= 1
        candidates = [token for token in counts if counts[token] > 0]
        prediction = min(candidates, key=lambda t: (-counts[t], t)) if candidates else -1
        result.append(prediction)
    return result


def summarize(rows):
    if not rows:
        raise ValueError("empty prediction dump")
    frequency = Counter(row["target"] for row in rows)
    predicted = Counter(row["prediction"] for row in rows)
    successes = Counter(row["target"] for row in rows if row["correct"])
    token_text = {row["target"]: row["target_text"] for row in rows}
    token_text.update({row["prediction"]: row["prediction_text"] for row in rows})
    correct = sum(successes.values())
    by_fact = defaultdict(list)
    for row in rows:
        by_fact[row["fact_index"]].append(row)
    baselines = {}
    for name, key in (("position", lambda row: row["position"]),
                      ("previous token", lambda row: row["input_token"]),
                      ("previous token + position", lambda row: (row["input_token"], row["position"]))):
        predictions = leave_one_out_majority(rows, key)
        baselines[name] = {
            "correct": sum(p == row["target"] for p, row in zip(predictions, rows)),
            "agrees_on_mlp_correct": sum(p == row["target"] and row["correct"]
                                         for p, row in zip(predictions, rows)),
            "abstained": predictions.count(-1),
        }
    best_tokens = [
        {"id": token, "text": token_text[token], "correct": count,
         "count": frequency[token], "recall": count / frequency[token],
         "predicted": predicted[token], "precision": count / predicted[token]}
        for token, count in successes.most_common()
    ]
    confidences = {}
    for label, subset in (("correct", [r for r in rows if r["correct"]]),
                          ("wrong", [r for r in rows if not r["correct"]])):
        confidences[label] = {
            field: statistics.median(row[field] for row in subset)
            for field in ("top1_probability", "target_probability", "target_rank", "margin", "hidden_norm")
        } if subset else {}
    examples = [
        {field: row[field] for field in ("fact_index", "position", "prefix", "target_text",
                                         "prediction_text", "correct", "target_probability", "target_rank")}
        for row in rows if row["position"] == 4
    ]
    prefix_correct = Counter()
    for fact_rows in by_fact.values():
        count = 0
        for row in fact_rows:
            if not row["correct"]:
                break
            count += 1
        prefix_correct[count] += 1
    outcome = {
        "overall": metrics(rows), "facts": len(by_fact),
        "source_correct": sum(row["source_prediction"] == row["target"] for row in rows),
        "fresh_correct": sum(row["fresh_prediction"] == row["target"] for row in rows),
        "fresh_and_trained_correct": sum(row["fresh_prediction"] == row["target"] and row["correct"] for row in rows),
        "target_types": len(frequency), "predicted_types": len(predicted),
        "ever_correct_target_types": len(successes),
        "zero_correct_target_types": len(frequency.keys() - successes.keys()),
        "best_constant_prediction_correct": max(frequency.values()),
        "without_eos_or_punctuation": metrics([r for r in rows if r["category"] not in ("EOS", "Punctuation")]),
        "by_category": group_metrics(rows, lambda row: row["category"]),
        "by_target_frequency": group_metrics(rows, lambda row: frequency_bin(frequency[row["target"]])),
        "by_position": group_metrics(rows, lambda row: row["position"] + 1),
        "first_generated_token": metrics([r for r in rows if r["position"] == 4]),
        "complete_teacher_forced_facts": sum(all(r["correct"] for r in group) for group in by_fact.values()),
        "leading_correct_tokens_per_fact": dict(sorted(prefix_correct.items())),
        "baselines": baselines, "confidence_medians": confidences,
        "top_correct_tokens": best_tokens,
        "top_predictions": [{"id": token, "text": token_text[token], "predicted": count,
                              "correct": successes[token], "target_count": frequency[token]}
                             for token, count in predicted.most_common()],
        "correct_covered_by_top10_tokens": sum(t["correct"] for t in best_tokens[:10]) / correct if correct else 0,
        "first_token_examples": examples,
    }
    return outcome


def table(headers, rows):
    escape = lambda cell: html.escape(str(cell))
    return "<table><thead><tr>" + "".join(f"<th>{escape(h)}</th>" for h in headers) + "</tr></thead><tbody>" + "".join(
        "<tr>" + "".join(f"<td>{escape(cell)}</td>" for cell in row) + "</tr>" for row in rows) + "</tbody></table>"


def render_report(summary, rows):
    pct = lambda value: f"{100 * value:.2f}%"
    total = summary["overall"]
    sections = ["<h1>Which A3 readout predictions are correct?</h1>",
                f"<p>{total['correct']:,}/{total['count']:,} correct ({pct(total['accuracy'])}) on "
                f"{summary['facts']:,} training facts. All predictions are teacher-forced suffix/EOS targets "
                "after five supplied tokens. This is training-set analysis, not held-out accuracy.</p>",
                "<h2>Token categories</h2><p>Categories are a fixed lexical heuristic, not semantic annotations. "
                "A correct EOS or punctuation prediction is useful, but does not alone establish factual recall.</p>",
                table(["Category", "Correct", "Targets", "Accuracy", "Share of successes"],
                      [(name, m['correct'], m['count'], pct(m['accuracy']), pct(m['correct'] / total['correct'] if total['correct'] else 0))
                       for name, m in summary['by_category'].items()]),
                "<h2>Target frequency in scored corpus</h2>",
                table(["Occurrences of target ID", "Correct", "Targets", "Accuracy"],
                      [(key, summary['by_target_frequency'][key]['correct'], summary['by_target_frequency'][key]['count'],
                        pct(summary['by_target_frequency'][key]['accuracy']))
                       for key in ("1", "2", "3-5", "6-10", "11-20", "21-50", "51-100", "101+")
                       if key in summary['by_target_frequency']]),
                "<h2>Most frequent correct target tokens</h2>",
                table(["Token text", "Correct / occurrences", "Recall", "Times predicted", "Precision"],
                      [(repr(t['text']), f"{t['correct']}/{t['count']}", pct(t['recall']), t['predicted'], pct(t['precision']))
                       for t in summary['top_correct_tokens'][:50]]),
                "<h2>Target position (zero-based)</h2>",
                table(["Target index (zero-based)", "Correct", "Targets", "Accuracy"],
                      [(key, m['correct'], m['count'], pct(m['accuracy'])) for key, m in sorted(summary['by_position'].items())]),
                "<h2>Simple leave-one-row-out majority baselines</h2>",
                "<p>Each row's own target is excluded. Ties choose the lowest token ID; unseen groups abstain. "
                "Other rows of the same fact remain eligible, so these are descriptive controls, not held-out baselines.</p>",
                table(["Condition", "Correct", "Accuracy", "Overlap with MLP successes", "Abstentions"],
                      [(name, b['correct'], pct(b['correct'] / total['count']), b['agrees_on_mlp_correct'], b['abstained'])
                       for name, b in summary['baselines'].items()]),
                "<h2>All scored predictions</h2><p>Search using the browser's Find command. Fact and input position are zero-based; "
                "the predicted target follows the displayed prefix. Probabilities come from the fitted readout.</p>",
                table(["Fact", "Input position", "Prefix", "Target", "Prediction", "Correct", "P(target)", "Target rank"],
                      [(r['fact_index'], r['position'], r['prefix'], repr(r['target_text']), repr(r['prediction_text']),
                        'yes' if r['correct'] else 'no', pct(r['target_probability']), r['target_rank']) for r in rows])]
    return "<!doctype html><html><meta charset='utf-8'><title>A3 puzzle correct predictions</title>" \
        "<style>body{font:15px/1.5 system-ui;margin:32px;color:#172336;background:#f5f7fa}" \
        "table{border-collapse:collapse;background:white;margin:12px 0 30px;max-width:100%}" \
        "td,th{padding:6px 12px;border:1px solid #dfe4eb;text-align:left}th{background:#e8eef5}" \
        "p{max-width:1050px}h2{margin-top:34px}td{white-space:pre-wrap}</style><body>" + "".join(sections) + "</body></html>"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    rows = load_rows(args.predictions)
    summary = summarize(rows)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "report.html").write_text(render_report(summary, rows))
    print(json.dumps({k: v for k, v in summary.items() if k not in ('top_correct_tokens', 'top_predictions', 'first_token_examples')}, indent=2))


if __name__ == "__main__":
    main()
