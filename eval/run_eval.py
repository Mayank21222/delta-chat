#!/usr/bin/env python3
"""
Eval harness. Runs the real pipeline (ingest -> delta -> report -> index)
against every labeled pair in eval/datasets/, scores delta P/R/F1, then runs
the QA ground truth through grounded chat and scores groundedness +
correctness. Prints a scorecard. One command: `python3 eval/run_eval.py`
(or `make eval`).
"""
from __future__ import annotations

import glob
import json
import sys

sys.path.insert(0, ".")

from src.chat.answer import ask
from src.chat.llm import get_llm_client
from src.observability.logging import get_logger, log, new_correlation_id
from src.observability.tracing import Trace
from src.pipeline import run_pipeline
from eval.metrics import score_chat, score_delta

logger = get_logger("eval")


def run_one(dataset_path: str) -> dict:
    with open(dataset_path) as f:
        ds = json.load(f)

    result = run_pipeline(
        pid_a=ds["pid_a"]["pid"], path_a=ds["pid_a"]["path"], rev_a=ds["pid_a"]["revision"],
        pid_b=ds["pid_b"]["pid"], path_b=ds["pid_b"]["path"], rev_b=ds["pid_b"]["revision"],
        out_dir=f"output/{ds['pair_id']}",
    )

    system_entries = [e.to_dict() for e in result.delta_entries]
    delta_score = score_delta(system_entries, ds["ground_truth_delta"])

    llm = get_llm_client()
    correlation_id = new_correlation_id()
    trace = Trace(correlation_id, request_type="eval_chat")
    answers = []
    for qa in ds["qa"]:
        log(logger, "info", f"asking: {qa['question']}", correlation_id, stage="eval_chat")
        answers.append(ask(qa["question"], result.index, llm, trace))
    trace.finish_and_save()

    chat_score = score_chat(answers, ds["qa"])

    return {
        "pair_id": ds["pair_id"],
        "delta_score": delta_score,
        "chat_score": chat_score,
        "num_delta_entries": len(system_entries),
        "num_gt_entries": len(ds["ground_truth_delta"]),
    }


def print_scorecard(results: list[dict]) -> None:
    print("\n" + "=" * 72)
    print("EVAL SCORECARD")
    print("=" * 72)
    for r in results:
        d = r["delta_score"]
        c = r["chat_score"]
        print(f"\nPair: {r['pair_id']}")
        print(f"  Delta   -- P: {d.precision:.2f}  R: {d.recall:.2f}  F1: {d.f1:.2f}  "
              f"(TP={d.true_positives} FP={d.false_positives} FN={d.false_negatives})")
        if d.unmatched_gt:
            print(f"  Delta MISSES: {d.unmatched_gt}")
        print(f"  Chat    -- groundedness: {c.groundedness:.2f}  correctness: {c.correctness:.2f}  "
              f"refusal_accuracy: {c.refusal_accuracy:.2f}")
        for pq in c.per_question:
            status = "OK" if pq["keyword_match"] and pq["grounded"] else "FAIL"
            print(f"    [{status}] {pq['question'][:70]}")
            if status == "FAIL":
                print(f"           grounded={pq['grounded']} keyword_match={pq['keyword_match']} "
                      f"cited={pq['cited']}")
                print(f"           answer: {pq['answer'][:150]}")
    print("\n" + "=" * 72)


def main():
    dataset_paths = sorted(glob.glob("eval/datasets/*.json"))
    if not dataset_paths:
        print("No datasets found in eval/datasets/")
        sys.exit(1)

    results = [run_one(p) for p in dataset_paths]
    print_scorecard(results)


if __name__ == "__main__":
    main()
