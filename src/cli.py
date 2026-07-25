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
import time

sys.path.insert(0, ".")

from src.chat.answer import ask
from src.chat.llm import get_llm_client
from src.observability.logging import get_logger, log, new_correlation_id
from src.observability.tracing import Trace
from src.pipeline import run_pipeline

logger = get_logger("cli", quiet=True)


def type_text(text: str, delay: float = 0.03) -> None:
    """Print text character by character for a more interactive feel."""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        if char in ".!?\n":
            time.sleep(delay * 4)
        elif char == ",":
            time.sleep(delay * 3)
        elif char == " ":
            time.sleep(delay * 0.5)
        else:
            time.sleep(delay)
    print()


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
    type_text(f"Ingested {ds['pid_a']['pid']} ({sum(len(p.elements) for p in result.doc_a.pages)} elements)", delay=0.02)
    type_text(f"Ingested {ds['pid_b']['pid']} ({sum(len(p.elements) for p in result.doc_b.pages)} elements)", delay=0.02)
    type_text(f"Delta: {len(result.delta_entries)} changes", delay=0.02)
    type_text(f"Report: {result.report_md_path}", delay=0.02)
    print()
    return result


def is_greeting(q: str) -> bool:
    """Check if the input is a greeting or generic word."""
    greetings = {"hello", "hi", "hey", "ok", "okay", "yes", "no", "clear", "thanks", "thank you", "bye", "quit", "exit", "help", "?", "!"}
    return q.lower().strip() in greetings


def split_questions(text: str) -> list[str]:
    """Split multi-question input into individual questions."""
    import re
    # Split on newlines first
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    questions = []
    for line in lines:
        # Strip numbered prefixes like "1. " or "2) "
        cleaned = re.sub(r'^\d+[\.\)]\s*', '', line)
        if cleaned:
            questions.append(cleaned)
    if not questions:
        return [text.strip()]
    return questions


def interactive_chat(result):
    llm = get_llm_client()
    ds = load_pair("eval/datasets/export_gas_compressor_pair.json")
    print()
    type_text("=" * 60, delay=0.003)
    type_text("  Document Delta & Grounded Chat", delay=0.04)
    type_text("  Compare P&ID revisions. Ask questions. Get cited answers.", delay=0.02)
    type_text("=" * 60, delay=0.003)
    print()
    type_text("  Documents loaded:", delay=0.03)
    type_text(f"    PID A: {ds['pid_a']['pid']} ({ds['pid_a']['revision']})", delay=0.03)
    type_text(f"    PID B: {ds['pid_b']['pid']} ({ds['pid_b']['revision']})", delay=0.03)
    print()
    type_text(f"  Delta: {len(result.delta_entries)} changes detected", delay=0.03)
    type_text(f"  Index: {len(result.index.chunks)} searchable chunks", delay=0.03)
    type_text(f"  LLM:   {llm.__class__.__name__}", delay=0.03)
    print()
    type_text("  Commands: type 'quit' to exit, 'help' for examples", delay=0.03)
    type_text("=" * 60, delay=0.003)
    print()

    while True:
        try:
            raw = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not raw or raw.lower() in ("exit", "quit"):
            break

        # Split multi-question input into individual questions
        questions = split_questions(raw)

        for q in questions:
            if not q:
                continue

            # Handle greetings and generic words
            if is_greeting(q):
                responses = {
                    "hello": "Hello! Ask me anything about the two P&ID revisions.",
                    "hi": "Hi! I can help you compare two P&ID revisions. What would you like to know?",
                    "hey": "Hey! What would you like to know about the documents?",
                    "ok": "Ready for your question.",
                    "okay": "Ready for your question.",
                    "yes": "Ready for your question.",
                    "no": "Okay. Feel free to ask something else.",
                    "clear": "Screen cleared. What would you like to know?",
                    "thanks": "You're welcome! Anything else?",
                    "thank you": "You're welcome! Feel free to ask more questions.",
                    "bye": "Goodbye!",
                    "help": "Try asking:\n  - What changed about the PSV 9027B relief valve?\n  - What is the high-high alarm setpoint for PIT 9023?\n  - Was the mechanical interlock note removed in Rev B?",
                    "?": "I answer questions about two P&ID revisions. Try asking what changed between them!",
                }
                print()
                type_text(f"  {responses.get(q.lower(), 'Ready for your question.')}", delay=0.03)
                print()
                continue

            correlation_id = new_correlation_id()
            trace = Trace(correlation_id, request_type="interactive_chat")
            log(logger, "info", f"question: {q}", correlation_id, stage="chat")
            try:
                answer = ask(q, result.index, llm, trace)
            except RuntimeError as exc:
                print()
                type_text(f"  [error] {exc}", delay=0.03)
                print()
                trace.finish_and_save()
                continue
            trace_path = trace.finish_and_save()

            # Clean answer
            text = answer.answer_text.strip()
            if "Question:" in text:
                text = text.split("Question:")[0].strip()
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            if lines:
                text = lines[-1] if len(lines) > 1 else lines[0]

            print()
            type_text(f"  {text}", delay=0.03)
            print()
            if answer.cited_chunk_ids:
                type_text(f"  Sources: {', '.join(answer.cited_chunk_ids)}", delay=0.02)
            type_text(f"  Trace:   {trace_path}", delay=0.02)
            print()


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
