"""
Orchestrates ingest -> delta -> report -> (index build), wrapped in tracing
and structured logging. This is the one place both the CLI and the eval
harness call into, so "one documented command runs ingest -> report -> chat"
and the eval harness are guaranteed to exercise the same code path.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.canonical.model import CanonicalDocument
from src.chat.index import RetrievalIndex
from src.delta.engine import DeltaEntry, compute_delta
from src.delta.report import write_report
from src.ingest.base import ingest_pid
from src.observability.logging import get_logger, log, new_correlation_id
from src.observability.tracing import Trace

logger = get_logger("pipeline")


@dataclass
class PipelineResult:
    doc_a: CanonicalDocument
    doc_b: CanonicalDocument
    delta_entries: list[DeltaEntry]
    index: RetrievalIndex
    report_json_path: str
    report_md_path: str
    correlation_id: str
    trace_path: str


def run_pipeline(pid_a: str, path_a: str, rev_a: str,
                  pid_b: str, path_b: str, rev_b: str,
                  out_dir: str = "output") -> PipelineResult:
    correlation_id = new_correlation_id()
    trace = Trace(correlation_id, request_type="ingest_delta_report")
    log(logger, "info", "pipeline started", correlation_id, stage="start",
        pid_a=pid_a, pid_b=pid_b)

    with trace.span("ingest_pid_a", pid=pid_a):
        doc_a = ingest_pid(pid_a, path_a, rev_a)
        log(logger, "info", f"ingested {pid_a}: {sum(len(p.elements) for p in doc_a.pages)} elements",
            correlation_id, stage="ingest_pid_a")

    with trace.span("ingest_pid_b", pid=pid_b):
        doc_b = ingest_pid(pid_b, path_b, rev_b)
        log(logger, "info", f"ingested {pid_b}: {sum(len(p.elements) for p in doc_b.pages)} elements",
            correlation_id, stage="ingest_pid_b")

    with trace.span("compute_delta"):
        entries = compute_delta(doc_a, doc_b)
        log(logger, "info", f"delta: {len(entries)} entries", correlation_id, stage="compute_delta")

    with trace.span("write_report"):
        json_path, md_path = write_report(entries, pid_a, pid_b, out_dir)

    with trace.span("build_index"):
        index = RetrievalIndex()
        index.add_document(doc_a, "pid_a")
        index.add_document(doc_b, "pid_b")
        index.add_delta_report(entries)
        index.build()
        log(logger, "info", f"index built: {len(index.chunks)} chunks", correlation_id, stage="build_index")

    trace_path = trace.finish_and_save()
    log(logger, "info", "pipeline finished", correlation_id, stage="done", trace_path=trace_path)

    return PipelineResult(
        doc_a=doc_a, doc_b=doc_b, delta_entries=entries, index=index,
        report_json_path=json_path, report_md_path=md_path,
        correlation_id=correlation_id, trace_path=trace_path,
    )
