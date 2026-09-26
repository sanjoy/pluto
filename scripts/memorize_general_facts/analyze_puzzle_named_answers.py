#!/usr/bin/env python3
"""Score predeclared named-answer spans in the exact puzzle prediction dump.

The annotations were selected from corpus text before inspecting predictions.
They are narrow illustrative cohorts, not an exhaustive semantic classification.
Every measurement uses true preceding tokens (teacher forcing) on training data.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path

from analyze_puzzle_predictions import load_rows


NAMED_ANSWERS = {
    26: "deposition",
    42: "photosphere",
    77: "cashew apple",
    126: "phonetics",
    153: "The Cherry Orchard",
    324: "Orion's Belt",
    339: "lava",
    572: "meteorite",
    654: "garbanzo beans",
    719: "magma",
    797: "Istanbul",
    859: "focus",
    970: "Mona Lisa",
}

# All 34 statements containing the whole word "capital": city, country/empire.
# These 68 annotations include names inside the prompt; eligibility is computed
# from the dump rather than selected using model correctness. Gloss entities
# such as Casablanca, Toronto, and the Seine are deliberately not included.
CAPITAL_RELATIONS = {
    15: ("Ulaanbaatar", "Mongolia"),
    19: ("Oslo", "Norway"),
    66: ("Dublin", "Ireland"),
    79: ("Paris", "France"),
    83: ("Hanoi", "Vietnam"),
    121: ("Nairobi", "Kenya"),
    169: ("Rabat", "Morocco"),
    208: ("Buenos Aires", "Argentina"),
    264: ("Stockholm", "Sweden"),
    301: ("Ottawa", "Canada"),
    305: ("London", "United Kingdom"),
    341: ("Wellington", "New Zealand"),
    405: ("Athens", "Greece"),
    410: ("Lima", "Peru"),
    416: ("Dakar", "Senegal"),
    427: ("Prague", "Czechia"),
    441: ("Seoul", "South Korea"),
    447: ("Santiago", "Chile"),
    479: ("Copenhagen", "Denmark"),
    490: ("Accra", "Ghana"),
    536: ("Lisbon", "Portugal"),
    551: ("Kathmandu", "Nepal"),
    571: ("Constantinople", "Byzantine Empire"),
    625: ("Cusco", "Inca Empire"),
    643: ("Bangkok", "Thailand"),
    679: ("Bogota", "Colombia"),
    680: ("Warsaw", "Poland"),
    690: ("Madrid", "Spain"),
    693: ("Canberra", "Australia"),
    759: ("Reykjavik", "Iceland"),
    890: ("Rome", "Italy"),
    915: ("Helsinki", "Finland"),
    980: ("Vienna", "Austria"),
    1002: ("Budapest", "Hungary"),
}


def inspect_span(fact_index, answer, role, cohort, facts, by_fact):
    fact = facts[fact_index]
    occurrences = fact.count(answer)
    if fact_index == 915 and answer == "Finland":
        # "Helsinki, on the Gulf of Finland, is Finland's capital."
        # Select the capital relation, not the preceding geographic gloss.
        if occurrences != 2:
            raise ValueError("fact 915 must contain both Finland mentions")
        start = fact.rindex(answer)
    else:
        if occurrences != 1:
            raise ValueError(
                f"fact {fact_index}: expected one occurrence of {answer!r}")
        start = fact.index(answer)
    end = start + len(answer)
    selected = [
        row for row in by_fact[fact_index]
        if row["target_text"] != "<|endoftext|>"
        and len(row["prefix"]) < end
        and len(row["prefix"]) + len(row["target_text"]) > start
    ]
    first = [
        row for row in selected
        if len(row["prefix"]) <= start
        < len(row["prefix"]) + len(row["target_text"])
    ]
    if len(first) > 1:
        raise ValueError(f"overlapping target pieces in fact {fact_index}")
    fields = ("position", "prefix", "target", "prediction", "target_text",
              "prediction_text", "correct", "target_rank",
              "target_probability")
    return {
        "cohort": cohort,
        "role": role,
        "fact_index": fact_index,
        "fact": fact,
        "answer": answer,
        "character_start": start,
        "character_end": end,
        "historical_capital": cohort == "capital_entities"
        and fact_index in (571, 625),
        "eligible": bool(selected),
        "scored_pieces": len(selected),
        "correct_pieces": sum(row["correct"] for row in selected),
        "whole_scored_span_correct": (
            all(row["correct"] for row in selected) if selected else None),
        "actual_first_piece_scored": bool(first),
        "actual_first_piece_correct": first[0]["correct"] if first else None,
        "first_scored_piece_correct": (
            selected[0]["correct"] if selected else None),
        "partially_prompted": bool(selected) and not first,
        "pieces": [{field: row[field] for field in fields} for row in selected],
    }


def summarize_spans(spans):
    eligible = [span for span in spans if span["eligible"]]
    fully_scored = [span for span in eligible
                    if span["actual_first_piece_scored"]]
    return {
        "declared_spans": len(spans),
        "excluded_zero_scored_spans": len(spans) - len(eligible),
        "eligible_spans": len(eligible),
        "eligible_facts": len({span["fact_index"] for span in eligible}),
        "correct_whole_scored_spans": sum(
            span["whole_scored_span_correct"] for span in eligible),
        "scored_pieces": sum(span["scored_pieces"] for span in eligible),
        "correct_pieces": sum(span["correct_pieces"] for span in eligible),
        "correct_first_scored_pieces": sum(
            span["first_scored_piece_correct"] for span in eligible),
        "eligible_actual_first_pieces": len(fully_scored),
        "correct_actual_first_pieces": sum(
            span["actual_first_piece_correct"] for span in fully_scored),
        "correct_fully_scored_spans": sum(
            span["whole_scored_span_correct"] for span in fully_scored),
        "partially_prompted_spans": len(eligible) - len(fully_scored),
    }


def analyze(rows):
    by_fact = defaultdict(list)
    for row in rows:
        by_fact[row["fact_index"]].append(row)
    facts = {row["fact_index"]: row["prefix"] for row in rows
             if row["target_text"] == "<|endoftext|>"}
    for fact_index, fact_rows in by_fact.items():
        positions = [row["position"] for row in fact_rows]
        if positions != list(range(4, positions[-1] + 1)):
            raise ValueError(f"missing scored rows in fact {fact_index}")
        if fact_rows[-1]["target_text"] != "<|endoftext|>":
            raise ValueError(f"missing final EOS row in fact {fact_index}")
    for fact_index in NAMED_ANSWERS.keys() | CAPITAL_RELATIONS.keys():
        if fact_index not in facts:
            raise ValueError(f"missing annotated fact {fact_index}")
    named = [inspect_span(index, answer, "named_answer", "named_answers",
                          facts, by_fact)
             for index, answer in NAMED_ANSWERS.items()]
    capitals = [inspect_span(index, answer, role, "capital_entities",
                             facts, by_fact)
                for index, answers in CAPITAL_RELATIONS.items()
                for answer, role in zip(answers, ("city", "country_or_empire"))]
    cohorts = {
        "named_answers": named,
        "capital_entities": capitals,
        "capital_cities": [span for span in capitals if span["role"] == "city"],
        "capital_countries_or_empires": [span for span in capitals
                                        if span["role"] == "country_or_empire"],
    }
    return {
        "teacher_forced": True,
        "evaluation_scope": "training corpus; five supplied prompt tokens",
        "method": (
            "Predeclared exact answer-text spans are aligned to decoded target "
            "pieces by character overlap. Zero-scored spans are excluded from "
            "accuracy denominators. Actual first-piece accuracy excludes "
            "answers whose first piece was supplied in the prompt. Whole "
            "scored-span accuracy requires every scored piece to be correct; "
            "it is not a free-running generation result."),
        "scored_rows": len(rows),
        "correct_rows": sum(row["correct"] for row in rows),
        "summary": {name: summarize_spans(spans)
                    for name, spans in cohorts.items()},
        "spans": named + capitals,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path, help="prediction-probe TSV")
    parser.add_argument("--output", type=Path,
                        help="write JSON to this path instead of stdout")
    args = parser.parse_args()
    report = analyze(load_rows(args.predictions))
    report["predictions_path"] = str(args.predictions)
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
