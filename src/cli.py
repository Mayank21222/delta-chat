#!/usr/bin/env python3
"""
CLI entrypoint. Usage:

  python3 -m src.cli run --pair eval/datasets/export_gas_compressor_pair.json
      Ingests both PIDs, computes the delta, writes the report, then drops
      into an interactive grounded chat over both PIDs + the delta report.

  python3 -m src.cli report --pair eval/datasets/export_gas_compressor_pair.json
      Same, but exits after writing the report (no chat) -- useful for CI.

  python3 -m src.cli chat --pair eval/datasets/export_gas_compressor_pair.json -q "..."
      Runs the pipeline and asks a single question non-interactively.

See Makefile for the wrapped `make run` / `make chat` shortcuts.
"""
from __future__ import annotations

import argparse
import json
import sys

sys.path.insert(0, ".")

from src.chat.answer import ask
from src.chat.llm import get_llm_client
from src.observability.logging import get_logger, log, new_correlation_id
from src.observability.tracing import Trace
from src.pipeline import run_pipeline

logger = get_logger("cli")


def load_pair(pair_path: str):
    with open(pair_path) as f:
        return json.load(f)


def do_ingest_and_report(pair_path: str):
    ds = load_pair(pair_path)
    result = run_pipeline(
        pid_a=ds["pid_a"]["pid"], path_a=ds["pid_a"]["path"], rev_a=ds["pid_a"]["revision"],
        pid_b=ds["pid_b"]["pid"], path_b=ds["pid_b"]["path"], rev_b=ds["pid_b"]["revision"],
        out_dir=f"output/{ds['pair_id']}",
    )
    print(f"\nIngested: {ds['pid_a']['pid']} ({sum(len(p.elements) for p in result.doc_a.pages)} elements), "
          f"{ds['pid_b']['pid']} ({sum(len(p.elements) for p in result.doc_b.pages)} elements)")
    print(f"Delta: {len(result.delta_entries)} changes -> {result.report_md_path}")
    print(f"Trace: {result.trace_path}")
    print(f"Correlation id: {result.correlation_id}\n")
    return result


def interactive_chat(result):
    llm = get_llm_client()
    print(f"Grounded chat ready ({len(result.index.chunks)} indexed chunks, "
          f"LLM_PROVIDER={llm.__class__.__name__}). Type a question, or 'exit'.\n")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q or q.lower() in ("exit", "quit"):
            break
        correlation_id = new_correlation_id()
        trace = Trace(correlation_id, request_type="interactive_chat")
        log(logger, "info", f"question: {q}", correlation_id, stage="chat")
        try:
            answer = ask(q, result.index, llm, trace)
        except RuntimeError as exc:
            print(f"[error] {exc}\n")
            trace.finish_and_save()
            continue
        trace_path = trace.finish_and_save()
        print(f"\n{answer.answer_text}\n")
        grounded_status = "Yes" if answer.grounded else "No"
        print(f"[cited: {answer.cited_chunk_ids or 'none'} | grounded: {grounded_status} | "
              f"trace: {trace_path}]\n")


def do_single_question(result, question: str):
    llm = get_llm_client()
    correlation_id = new_correlation_id()
    trace = Trace(correlation_id, request_type="single_chat")
    answer = ask(question, result.index, llm, trace)
    trace_path = trace.finish_and_save()
    print(f"\nQ: {question}\nA: {answer.answer_text}\n")
    print(f"[cited: {answer.cited_chunk_ids or 'none'} | grounded: {answer.grounded} | trace: {trace_path}]")


def main():
    parser = argparse.ArgumentParser(description="Document Delta & Grounded Chat")
    parser.add_argument("command", choices=["run", "report", "chat"])
    parser.add_argument("--pair", required=True, help="path to a pair json (see eval/datasets/)")
    parser.add_argument("-q", "--question", help="single question for `chat` command")
    args = parser.parse_args()

    result = do_ingest_and_report(args.pair)

    if args.command == "report":
        return
    if args.command == "chat" and args.question:
        do_single_question(result, args.question)
        return
    interactive_chat(result)


if __name__ == "__main__":
    main()
