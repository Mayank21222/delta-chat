# Demo

## One-Command Pipeline

```bash
$ make run
```

```
Ingesting PID A (pid-export-a)... 695 elements extracted
Ingesting PID B (pid-export-b)... 695 elements extracted
Computing delta... 7 changes detected
Writing report... output/export_gas_compressor_A_vs_B/delta_report.md
Indexing for chat... 1400 chunks ready
```

## Delta Report

The delta engine produces a structured, human-readable report:

```markdown
# Delta Report: pid-export-a -> pid-export-b

**Summary:** 7 changes — 2 added, 2 removed, 3 modified.

## Modified (3)
- [D0001] Changed: "SP= 225.4 bar (g)" -> "SP= 230.0 bar (g)"
  type: setpoint · location: sheet 1 (1035, 68) · confidence: 0.89
- [D0002] Changed: "HH : 214" -> "HH : 220"
  type: setpoint · location: sheet 1 (676, 135) · confidence: 0.84
- [D0003] Changed: "1835 NOTE 29" -> "NOTE 29"
  type: text · location: sheet 1 (183, 569) · confidence: 0.83

## Removed (2)
- [D0004] Removed: "MECHANICAL"
- [D0005] Removed: "INTERLOCK"

## Added (2)
- [D0006] Added: "1902"
- [D0007] Added: "36. REV B: SUCTION STRAINER UPGRADED TO PERMANENT TYPE."
```

## Grounded Chat

Ask questions about either document or the delta — answers carry citations back to source content:

```bash
>> What changed about the PSV 9027B relief valve setpoint?
```

```
The PSV 9027B relief valve setpoint was changed from 225.4 bar(g) to
230.0 bar(g) in Rev B. This is documented in delta entry D0001 as a
modified setpoint on sheet 1.
```

*[delta:D0001 | grounded: True]*

```bash
>> Was the mechanical interlock note removed in Rev B?
```

```
Yes, the mechanical interlock note was removed in Rev B. Delta entries
D0004 and D0005 show "MECHANICAL" and "INTERLOCK" were both removed.
```

*[delta:D0004, delta:D0005 | grounded: True]*

```bash
>> What is the compressor's serial number?
```

```
Not found in the provided documents.
```

*[grounded: True | refused correctly]*

## Eval Scorecard

```bash
$ make eval
```

```
Pair: export_gas_compressor_native_vs_scanned
  Delta   -- P: 0.00  R: 0.00  F1: 0.00  (TP=0 FP=693 FN=0)
  Chat    -- groundedness: 1.00  correctness: 0.00  refusal_accuracy: 1.00

Pair: export_gas_compressor_A_vs_B
  Delta   -- P: 1.00  R: 1.00  F1: 1.00  (TP=7 FP=0 FN=0)
  Chat    -- groundedness: 1.00  correctness: 0.40  refusal_accuracy: 1.00
```

- **Delta F1 = 1.0** on the primary pair (all 7 ground truth changes detected)
- **Groundedness = 1.0** (all citations map to retrieved chunks)
- Correctness under mock is low by construction (mock returns templated answers)
