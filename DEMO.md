# Demo walkthrough

*(Submitting as a written walkthrough with real captured output, in place of
a screen recording — swap in a 2-4 min recording of the same flow if you'd
rather show it live; the commands below are exactly what to run.)*

## 1. One delta

```bash
$ make run PAIR=eval/datasets/export_gas_compressor_pair.json
```

Ingests the real Rev A P&ID and the synthesized Rev B (see
`data/samples/synthesize_revb.py` for the exact, documented edits), computes
the delta, writes the report. Real captured output:

```
Ingested: pid-export-a (695 elements), pid-export-b (695 elements)
Delta: 7 changes -> output/export_gas_compressor_A_vs_B/delta_report.md
Trace: traces/<correlation_id>.json
Correlation id: <correlation_id>
```

Resulting report (`output/export_gas_compressor_A_vs_B/delta_report.md`, real
output, not hand-edited):

```markdown
# Delta Report: pid-export-a -> pid-export-b

**Summary:** 7 changes — 2 added, 2 removed, 3 modified.

## Modified (3)

### Sheet 1
- **[D0001]** Changed: "SP= 225.4 bar (g)" -> "SP= 230.0 bar (g)"
  type: setpoint · location: sheet 1 (1035, 68) · confidence: 0.89
- **[D0002]** Changed: "HH : 214" -> "HH : 220"
  type: setpoint · location: sheet 1 (676, 135) · confidence: 0.84
- **[D0003]** Changed: "1835 NOTE 29" -> "NOTE 29"
  type: text · location: sheet 1 (183, 569) · confidence: 0.83

## Added (2)
- **[D0006]** Added: "1902"
- **[D0007]** Added: "36. REV B: SUCTION STRAINER UPGRADED TO PERMANENT TYPE."

## Removed (2)
- **[D0004]** Removed: "MECHANICAL"
- **[D0005]** Removed: "INTERLOCK"
```

Note the split of the duty-figure change into D0003 (modified) + D0006
(added) rather than one clean "1835 -> 1902" entry — documented and explained
in README "Known limitations #1". Left in deliberately rather than curated
away, because it's the most honest illustration of "alignment is the hard
part" in this whole submission.

## 2. One grounded chat exchange

With `ollama serve` running and `llama3.1:8b` pulled:

```bash
$ make chat -- -q "What changed about the PSV 9027B relief valve setpoint?"
```

*[Paste your real Ollama output here before submitting — this needs a live
model. The mock-provider version below is what runs without Ollama, useful
to show the retrieval/citation plumbing works, but it is NOT a substitute
for a real answer.]*

```
Q: What changed about the PSV 9027B relief valve setpoint?
A: [real Ollama output goes here, expected to reference 225.4 -> 230.0
    bar(g) and cite delta:D0001]

[cited: [...] | grounded: True | trace: traces/<id>.json]
```

## 3. Eval scorecard

```bash
$ LLM_PROVIDER=mock python3 eval/run_eval.py
```

Real captured output (`eval/sample_scorecard_mock.txt`):

```
========================================================================
EVAL SCORECARD
========================================================================

Pair: export_gas_compressor_native_vs_scanned
  Delta   -- P: 0.00  R: 0.00  F1: 0.00  (TP=0 FP=693 FN=0)
  Chat    -- groundedness: 1.00  correctness: 0.00  refusal_accuracy: 1.00  retrieval_hit: 0.00
    [FAIL] What is the relief valve setpoint for PSV 9027B?
    [FAIL] Who manufactures the export compressor?
    [FAIL] What is the high-high alarm setpoint for PIT 9023?
    [FAIL] How many stages does the compressor have?
    [FAIL] What is the motor drive power rating?

Pair: export_gas_compressor_A_vs_B
  Delta   -- P: 1.00  R: 1.00  F1: 1.00  (TP=7 FP=0 FN=0)
  Chat    -- groundedness: 1.00  correctness: 0.40  refusal_accuracy: 1.00  retrieval_hit: 0.40
    [FAIL] What changed about the PSV 9027B relief valve setpoint between Rev A and Rev B?
    [FAIL] What is the high-high alarm setpoint for PIT 9023 in Rev B?
    [FAIL] Who is the vendor for the 3rd stage HP gas export compressor?
    [OK] Was the mechanical interlock note removed in Rev B?
    [OK] What is the compressor's serial number?

  (no regressions vs previous run)
========================================================================
```

Correctness is low under `LLM_PROVIDER=mock` **by construction** — the mock
LLM emits a templated non-answer so the harness is runnable without a live
model, and this run validates retrieval → citation → grounding-check
plumbing (groundedness: 1.00) rather than answer quality. Re-run with
`LLM_PROVIDER=ollama` (the real path) for real correctness numbers.
