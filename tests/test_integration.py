"""
Integration tests for the full pipeline (ingest -> delta -> report -> index).
No LLM required -- uses the mock provider for the chat layer. These tests
exercise the real PDFs in data/samples/raw/ to validate end-to-end behavior.
"""
import os
import sys
import json

sys.path.insert(0, ".")

from src.pipeline import run_pipeline
from src.chat.answer import ask
from src.chat.llm import MockLLMClient
from src.observability.tracing import Trace
from src.observability.logging import new_correlation_id
from eval.metrics import score_delta, score_chat


PAIR_A_PATH = "data/samples/raw/pid_export_gas_compressor_revA.pdf"
PAIR_B_PATH = "data/samples/raw/pid_export_gas_compressor_revB.pdf"
SCANNED_PATH = "data/samples/raw/pid_export_gas_compressor_revA_scanned.pdf"


def test_native_to_native_pipeline():
    """Full pipeline on native PDF pair -- the primary happy path."""
    result = run_pipeline(
        pid_a="test-a", path_a=PAIR_A_PATH, rev_a="Rev A",
        pid_b="test-b", path_b=PAIR_B_PATH, rev_b="Rev B",
        out_dir="/tmp/test_native_native",
    )
    assert len(result.delta_entries) == 7
    assert result.index is not None
    assert len(result.index.chunks) > 0
    assert os.path.exists(result.report_json_path)
    assert os.path.exists(result.report_md_path)
    assert os.path.exists(result.trace_path)


def test_native_to_native_delta_scores_perfectly():
    """Delta entries must match ground truth exactly."""
    ground_truth = [
        {"change_type": "modified", "must_contain": ["225.4", "230.0"], "note": "setpoint"},
        {"change_type": "modified", "must_contain": ["214", "220"], "note": "alarm"},
        {"change_type": "removed", "must_contain": ["MECHANICAL"], "note": "mechanical"},
        {"change_type": "removed", "must_contain": ["INTERLOCK"], "note": "interlock"},
        {"change_type": "modified", "must_contain": ["1835"], "note": "duty old"},
        {"change_type": "added", "must_contain": ["1902"], "note": "duty new"},
        {"change_type": "added", "must_contain": ["REV B", "STRAINER"], "note": "rev note"},
    ]
    result = run_pipeline(
        pid_a="test-a", path_a=PAIR_A_PATH, rev_a="Rev A",
        pid_b="test-b", path_b=PAIR_B_PATH, rev_b="Rev B",
        out_dir="/tmp/test_delta_scores",
    )
    system = [e.to_dict() for e in result.delta_entries]
    score = score_delta(system, ground_truth)
    assert score.f1 == 1.0, f"Expected perfect F1, got {score.f1} (TP={score.true_positives} FP={score.false_positives} FN={score.false_negatives})"


def test_scanned_pdf_ingestion():
    """Scanned PDF adapter produces a valid CanonicalDocument."""
    from src.ingest.base import ingest_pid
    doc = ingest_pid("test-scanned", SCANNED_PATH, "Rev A (scanned)")
    assert doc.source_format == "pdf_scanned"
    assert len(doc.pages) >= 1
    total_elements = sum(len(p.elements) for p in doc.pages)
    assert total_elements > 50, f"Expected >50 elements from scanned PDF, got {total_elements}"
    for el in doc.all_elements():
        assert el.source == "ocr"
        assert 0 <= el.confidence <= 1.0


def test_native_vs_scanned_cross_format():
    """Cross-format comparison: native vs scanned of same revision.
    Should produce deltas from OCR noise, but the pipeline should not crash."""
    result = run_pipeline(
        pid_a="test-native", path_a=PAIR_A_PATH, rev_a="Rev A",
        pid_b="test-scanned", path_b=SCANNED_PATH, rev_b="Rev A (scanned)",
        out_dir="/tmp/test_cross_format",
    )
    assert result.delta_entries is not None
    assert len(result.delta_entries) > 0
    assert result.index is not None


def test_grounded_chat_end_to_end():
    """Full pipeline + grounded chat with mock LLM."""
    result = run_pipeline(
        pid_a="test-a", path_a=PAIR_A_PATH, rev_a="Rev A",
        pid_b="test-b", path_b=PAIR_B_PATH, rev_b="Rev B",
        out_dir="/tmp/test_chat",
    )
    llm = MockLLMClient()
    correlation_id = new_correlation_id()
    trace = Trace(correlation_id, request_type="integration_test")

    answer = ask(
        "What changed about the PSV 9027B relief valve setpoint?",
        result.index, llm, trace,
    )
    trace.finish_and_save()

    assert answer.answer_text is not None
    assert len(answer.answer_text) > 0
    assert answer.cited_chunk_ids is not None
    assert answer.grounded is True


def test_retrieval_index_searches_all_sources():
    """Retrieval index should contain chunks from both PIDs and delta report."""
    result = run_pipeline(
        pid_a="test-a", path_a=PAIR_A_PATH, rev_a="Rev A",
        pid_b="test-b", path_b=PAIR_B_PATH, rev_b="Rev B",
        out_dir="/tmp/test_index",
    )
    index = result.index
    kinds = {c.kind for c in index.chunks}
    assert "pid_a" in kinds, "Index should contain PID A chunks"
    assert "pid_b" in kinds, "Index should contain PID B chunks"
    assert "delta" in kinds, "Index should contain delta report chunks"

    # Search should return results for a term that actually appears in the docs
    results = index.search("compressor", top_k=5)
    assert len(results) > 0, "Expected at least one result for 'compressor'"


def test_trace_has_all_spans():
    """A pipeline trace must include spans for every stage."""
    result = run_pipeline(
        pid_a="test-a", path_a=PAIR_A_PATH, rev_a="Rev A",
        pid_b="test-b", path_b=PAIR_B_PATH, rev_b="Rev B",
        out_dir="/tmp/test_trace",
    )
    with open(result.trace_path) as f:
        trace = json.load(f)
    span_names = [s["name"] for s in trace["spans"]]
    assert "ingest_pid_a" in span_names
    assert "ingest_pid_b" in span_names
    assert "compute_delta" in span_names
    assert "write_report" in span_names
    assert "build_index" in span_names
    assert trace["total_duration_ms"] > 0


if __name__ == "__main__":
    import subprocess
    sys.exit(subprocess.call(["python3", "-m", "pytest", __file__, "-v"]))
