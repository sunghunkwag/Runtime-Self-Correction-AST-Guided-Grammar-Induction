# Runtime Evolutionary Self-Correction via AST-Guided Grammar Induction

A single-file recursive self-improvement engine that evolves Python programs through AST analysis and statistical grammar learning (EDA).

## Core Capabilities

### 1. Unified Engine (v2.0)
- **EDA Grammar Learning**: Dynamically weights AST nodes based on elite solutions (e.g., 'var' vs 'const').
- **Meta-RSI Autopatch**: L0-L5 self-modification levels (Hyperparams to Operator Logic).
- **Test-Driven Repair (TDR)**: Atomic patching with rollback on syntax/verification failure.

### 2. ARC Gymnasium
- **Real Data**: Loads official Kaggle ARC JSON files.
- **Dynamic Discovery**: Scans `ARC_GYM` directory for tasks.
- **Evaluation**: Pixel-perfect grid matching and heuristic scoring.

### 3. Safety Architecture
- **Step Injection**: Prevents infinite loops via AST transformation.
- **Sandboxed Execution**: Whitelisted statements and built-ins only.
- **Surrogate Model**: Pre-filters candidates to save compute.

## Verification

Run the integrated test suite coverage (EDA, Algorithmic, ARC, TDR, Autopatch):

```bash
python test_suite.py
```

For full integration verification, run:

```bash
python verify_full_integration.py
```

## Performance Metrics

- **Grammar Adaptation**: Confirmed significant weight shifts (e.g., Var 2.0 -> 21.0) during evolution.
- **Safety**: 100% rollback rate on injected syntax errors.
- **Integration**: Coexistence of standard optimization (L2) and meta-learning (L5) verified.

## Hybrid Bridge

The `hybrid_bridge/` package provides a loose-coupled handoff between Omega Forge evidence
(JSONL) and UNIFIED_RSI_EXTENDED tuning. It does not merge the codebases; it reads Omega
outputs, tunes the next configuration, and writes `omega_config.json` for the next round.

Quick start (dry-run with existing evidence):

```bash
python -m hybrid_bridge.run_round --dry-run --out evidence.jsonl --max_generations 2000
```

Full round (runs Omega Forge as a subprocess):

```bash
python -m hybrid_bridge.run_round --omega-path OMEGA_FORGE_V13.py --target 6 --max_generations 2000
```

You can inspect `hybrid_bridge/schema.py` for the evidence/config dataclasses and run
`python -m hybrid_bridge.selftest_bridge` to validate parsing, tuning, and dry-run wiring.

## License

MIT License
