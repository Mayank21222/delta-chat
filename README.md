# Document Delta & Grounded Chat

**Automated comparison of engineering document revisions with AI-powered grounded Q&A.**

Given two revisions of a P&ID (PID A, PID B), this system ingests both regardless of format, computes a structured delta between them, produces a human-readable delta report, and lets you chat with both documents and the delta — with citations grounded in the actual source content.

Built as a take-home assignment for an Applied AI Engineer role. Demonstrates format-agnostic ingestion, deterministic delta computation, retrieval-augmented generation with citation enforcement, and full observability.

---

## Demo

```bash
# Run the full pipeline
make run

# Ask questions about the documents
>> What changed about the PSV 9027B relief valve setpoint?
  The setpoint was changed from 225.4 bar(g) to 230.0 bar(g) [delta:D0001].

>> Was the mechanical interlock note removed in Rev B?
  Yes, the mechanical interlock note was removed in Rev B [delta:D0004].

>> What is the compressor's serial number?
  Not found in the provided documents.
```

---

## Features

- **Format-agnostic ingestion** — Native PDF + Scanned PDF (OCR) behind a single adapter interface. DWG stubbed behind the same seam.
- **Deterministic delta engine** — 4-pass alignment (exact → fuzzy → moved → unmatched) with typed, located, confidence-scored entries. Zero LLM in the delta path.
- **Grounded chat** — TF-IDF retrieval over both PIDs + delta report. Every answer carries citations. Refuses when unsupported.
- **Observability** — Per-request JSON traces with stage timing, token/cost telemetry, structured logs with correlation IDs.
- **Evaluation harness** — Delta P/R/F1, chat groundedness/correctness/refusal accuracy, regression detection across runs.

---

## Quick Start

### Prerequisites

```bash
# Python dependencies
pip install -r requirements.txt

# System dependencies (macOS)
brew install tesseract poppler ollama

# Pull the LLM model
ollama pull llama3.1:8b
```

### Run

```bash
# Full pipeline: ingest → delta → report → interactive chat
make run

# Run eval scorecard
make eval

# Run tests
make test
```

### Configuration

Edit `.env` to switch between Ollama (real LLM) and mock (offline):

```bash
LLM_PROVIDER=ollama    # or "mock" for offline/deterministic mode
OLLAMA_MODEL=llama3.1:8b
OLLAMA_HOST=http://localhost:11434
```

---

## Architecture

```
PID (bytes+meta) --> FormatAdapter --> CanonicalDocument --> DeltaEngine --> DeltaEntry[]
                      (pdf_native /                                |
                       pdf_scanned /                                v
                       dwg stub)                              DeltaReport (MD+JSON)
                                                                     |
                      CanonicalDocument A, B  +  DeltaEntry[]  -->  RetrievalIndex (TF-IDF)
                                                                     |
                                            question --> retrieve --+--> LLM (Ollama) --> cited answer
```

### Key Design Decisions

**Canonical representation is the crux.** Every adapter produces the same shape: pages holding line-level elements with `(type, text, bbox, page, confidence, source)`. The delta engine, retrieval index, and chat layer never import an `ingest/*` module — they only see `CanonicalDocument`. Adding a 4th format costs one adapter class.

**Delta engine is 100% deterministic.** No LLM anywhere in `src/delta/`. The LLM appears exactly once, in `src/chat/llm.py`/`answer.py`, to answer over an already-computed delta. This makes the delta report byte-for-byte reproducible.

**Grounding is enforced mechanically.** The system prompt requires `[chunk_id]` citations on every claim. Citations are parsed back out and cross-checked against retrieved chunks — "did it cite something real" is a checkable fact, not a vibe.

---

## Tech Stack

- **Python 3.9+** — Core language
- **PyMuPDF** — Native PDF extraction
- **Tesseract OCR** — Scanned PDF processing
- **scikit-learn** — TF-IDF retrieval
- **Ollama** — Local LLM inference (swappable)
- **difflib** — Text similarity for alignment

---

## Project Structure

```
delta-chat/
├── src/
│   ├── ingest/          # Format adapters (pdf_native, pdf_scanned, dwg stub)
│   ├── canonical/       # Format-agnostic intermediate model
│   ├── delta/           # Alignment engine + delta computation + report generation
│   ├── chat/            # TF-IDF retrieval + LLM client + grounded answering
│   └── observability/   # Request tracing + structured logging
├── eval/
│   ├── datasets/        # Labeled document pairs with ground truth
│   ├── metrics.py       # Delta P/R/F1, chat groundedness/correctness
│   └── run_eval.py      # One-command eval harness
├── data/samples/        # Real P&IDs + synthesized revisions
└── tests/               # Unit + integration tests
```

---

## Eval Scorecard

```
make eval

Pair: export_gas_compressor_A_vs_B
  Delta   -- P: 1.00  R: 1.00  F1: 1.00  (TP=7 FP=0 FN=0)
  Chat    -- groundedness: 1.00  correctness: 0.40  refusal_accuracy: 1.00
```

- **Delta F1 = 1.0** on the primary pair (7/7 ground truth changes detected)
- **Groundedness = 1.0** (all citations map to retrieved chunks)
- **Honest failure:** Cross-format pair scores 693 false positives from OCR noise (P/R/F1 = 0.0)

---

## What I Cut

To keep scope tight for the take-home, I deliberately excluded:

- **Vector DB / embeddings** — TF-IDF works for 2-PID scope; embeddings needed at500+ sheets
- **Multi-page table alignment** — tables that span pages aren't aligned across pages
- **Semantic similarity for alignment** — pure text+spatial scoring; no sentence-transformers
- **Streaming API** — Ollama client collects full response; streaming only in CLI display
- **Incremental indexing** — full rebuild on every run; no delta-only updates
- **DWG parsing** — adapter stub exists behind a real seam; no ODA/ezdxf dependency
- **PDF merge/overlay** — delta visualization is text-only; no visual markup layer
- **Authentication / multi-tenant** — single-user CLI tool; no auth layer
- **CI/CD pipeline** — no GitHub Actions; eval is local-only

---

## What I'd Do Next

- **Embeddings + vector DB** for better retrieval at scale (500+ sheets)
- **Spatially-adjacent chunk merging** to fix label/value retrieval gaps
- **DWG support** via ODA File Converter + ezdxf
- **Markup overlay** — draw delta back onto the PDF as bounding boxes
- **Cost/latency analysis** — rollup script over trace files

---

## License

MIT
