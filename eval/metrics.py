"""
Eval metrics.

Delta P/R/F1: a ground-truth entry is "matched" by a system delta entry if
the change_type agrees and every substring in `must_contain` appears
(case-insensitive) somewhere in that system entry's before/after/description
text. This is intentionally entry-scoped (not corpus-scoped) -- if a
matching entry does not exist, it's a miss, full stop. That's what makes
this catch a regression: loosen or break alignment and recall drops
visibly, it doesn't get papered over by fuzzy corpus-wide credit.

Chat groundedness: fraction of answers where every citation the model
emitted maps to a chunk that was actually retrieved (see chat/answer.py).
This catches citation hallucination specifically (citing something real
but wrong is a separate, harder problem -- noted as a limitation).

Chat correctness: keyword-presence check against the expected answer.
Crude on purpose -- exact-match QA scoring for free-text answers is its
own research problem; keyword presence is the honest, cheap version of it,
and is documented as such rather than dressed up as more rigorous than it is.
"""
from __future__ import annotations

from dataclasses import dataclass


def _entry_text(entry: dict) -> str:
    parts = [entry.get("before_text") or "", entry.get("after_text") or "", entry.get("description") or ""]
    return " ".join(parts).upper()


@dataclass
class DeltaScore:
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int
    unmatched_gt: list[str]


def score_delta(system_entries: list[dict], ground_truth: list[dict]) -> DeltaScore:
    matched_gt = set()
    matched_system = set()

    for gi, gt in enumerate(ground_truth):
        must = [s.upper() for s in gt["must_contain"]]
        for si, entry in enumerate(system_entries):
            if si in matched_system:
                continue
            if entry["change_type"] != gt["change_type"]:
                continue
            text = _entry_text(entry)
            if all(s in text for s in must):
                matched_gt.add(gi)
                matched_system.add(si)
                break

    tp = len(matched_gt)
    fn = len(ground_truth) - tp
    fp = len(system_entries) - len(matched_system)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    unmatched = [ground_truth[i]["note"] for i in range(len(ground_truth)) if i not in matched_gt]
    return DeltaScore(precision, recall, f1, tp, fp, fn, unmatched)


@dataclass
class ChatScore:
    groundedness: float
    correctness: float
    refusal_accuracy: float
    per_question: list[dict]


def score_chat(answers: list, qa_ground_truth: list[dict]) -> ChatScore:
    per_question = []
    grounded_count = 0
    correct_count = 0
    refusal_correct = 0

    for ans, gt in zip(answers, qa_ground_truth):
        is_refusal = "not found in the provided documents" in ans.answer_text.lower()
        expect_refusal = gt.get("expect_refusal", False)
        refusal_ok = is_refusal == expect_refusal

        if expect_refusal:
            keyword_ok = is_refusal
        else:
            answer_upper = ans.answer_text.upper()
            keywords = gt.get("expected_keywords", [])
            keyword_ok = all(k.upper() in answer_upper for k in keywords) if keywords else True

        grounded_count += int(ans.grounded or is_refusal)
        correct_count += int(keyword_ok)
        refusal_correct += int(refusal_ok)

        per_question.append({
            "question": gt["question"],
            "grounded": bool(ans.grounded or is_refusal),
            "keyword_match": keyword_ok,
            "refusal_expected": expect_refusal,
            "refusal_observed": is_refusal,
            "cited": ans.cited_chunk_ids,
            "answer": ans.answer_text,
        })

    n = len(qa_ground_truth) or 1
    return ChatScore(
        groundedness=grounded_count / n,
        correctness=correct_count / n,
        refusal_accuracy=refusal_correct / n,
        per_question=per_question,
    )
