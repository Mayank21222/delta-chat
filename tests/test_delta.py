"""
Unit tests covering the deterministic core: canonical model, alignment,
delta engine, metrics. Deliberately no tests here hit an LLM or need
network -- those are covered by the eval harness against the real Ollama
setup, which is not appropriate for a fast/CI unit test run.
"""
import sys
sys.path.insert(0, ".")

from src.canonical.model import BBox, CanonicalDocument, CanonicalElement, CanonicalPage, ElementType
from src.delta.align import align, MatchKind
from src.delta.engine import compute_delta, ChangeType
from eval.metrics import score_delta


def _doc(pid, texts, page_no=1):
    elements = [
        CanonicalElement(id=f"{page_no}:{i}", type=ElementType.TEXT, text=t,
                          bbox=BBox(i * 10, 0, i * 10 + 5, 5), page=page_no)
        for i, t in enumerate(texts)
    ]
    return CanonicalDocument(pid=pid, source_format="pdf_native", revision_label="test",
                              pages=[CanonicalPage(number=page_no, width=100, height=100, elements=elements)])


def test_identical_docs_produce_no_delta():
    doc = _doc("a", ["26-KA-902", "SP= 225.4 bar (g)", "NOTE 1"])
    entries = compute_delta(doc, doc)
    assert entries == []


def test_added_element_detected():
    a = _doc("a", ["26-KA-902"])
    b = _doc("b", ["26-KA-902", "NEW NOTE HERE"])
    entries = compute_delta(a, b)
    assert len(entries) == 1
    assert entries[0].change_type == ChangeType.ADDED
    assert entries[0].after_text == "NEW NOTE HERE"


def test_removed_element_detected():
    a = _doc("a", ["26-KA-902", "OLD NOTE"])
    b = _doc("b", ["26-KA-902"])
    entries = compute_delta(a, b)
    assert len(entries) == 1
    assert entries[0].change_type == ChangeType.REMOVED
    assert entries[0].before_text == "OLD NOTE"


def test_modified_element_matched_by_similarity_and_position():
    a = _doc("a", ["SP= 225.4 bar (g)"])
    b = _doc("b", ["SP= 230.0 bar (g)"])
    entries = compute_delta(a, b)
    assert len(entries) == 1
    assert entries[0].change_type == ChangeType.MODIFIED
    assert entries[0].before_text == "SP= 225.4 bar (g)"
    assert entries[0].after_text == "SP= 230.0 bar (g)"
    assert entries[0].confidence > 0.45


def test_alignment_exact_match_is_free():
    a = _doc("a", ["UNCHANGED LINE"])
    b = _doc("b", ["UNCHANGED LINE"])
    matches = align(a, b)
    assert len(matches) == 1
    assert matches[0].kind == MatchKind.EXACT
    assert matches[0].score == 1.0


def test_score_delta_perfect_match():
    system = [{"change_type": "modified", "before_text": "225.4", "after_text": "230.0", "description": ""}]
    gt = [{"change_type": "modified", "must_contain": ["225.4", "230.0"], "note": "setpoint"}]
    score = score_delta(system, gt)
    assert score.precision == 1.0
    assert score.recall == 1.0
    assert score.f1 == 1.0


def test_score_delta_catches_a_regression():
    """If the delta engine stops emitting the change entirely, recall must drop."""
    system = []
    gt = [{"change_type": "modified", "must_contain": ["225.4", "230.0"], "note": "setpoint"}]
    score = score_delta(system, gt)
    assert score.recall == 0.0
    assert score.unmatched_gt == ["setpoint"]


if __name__ == "__main__":
    import subprocess
    sys.exit(subprocess.call(["python3", "-m", "pytest", __file__, "-v"]))
