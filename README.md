# Runtime Evolutionary Self-Correction via AST-Guided Grammar Induction

> **An autonomous evolutionary engine that learns to generate better code by statistically analyzing the syntax trees of elite solutions, enabling self-directed improvement without external supervision.**

[![Status](https://img.shields.io/badge/Status-Research%20Prototype-yellow)]()
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## 📊 Objective Performance Metrics

### Grammar Learning (EDA)
- **Initial Distribution**: Uniform (all operators weighted 1.0-2.0)
- **After 5 Generations**: 
  - `var` (variable reference): **2.0 → 13.07** (+554%)
  - `const` (constants): **2.0 → 6.59** (+229%)
  - `binop` (binary ops): **2.0 → 1.71** (-14%)
- **Interpretation**: Engine learned that simpler variable-based expressions dominate elite solutions

### ARC Benchmark (Abstract Reasoning)
- **Task**: `arc_ident` (3×3 grid identity mapping)
  - **Result**: Solved in **Gen 1** with holdout error **0.0000**
- **Task**: `arc_25d8a9c8` (real Kaggle ARC JSON)
  - **Initial Score**: 3.65 (non-random, data loading verified)
  - **Holdout Error**: 4.40 pixels/grid

### Test-Driven Repair (Safety)
- **Syntax Error Injection**: 
  - Rollback success rate: **100%** (malformed patches rejected)
- **Valid Patch Acceptance**: 
  - Import verification: **PASS** (regression test executed)

---

## 🏗️ System Architecture

### Core Components

| Module | Function | Implementation |
|--------|----------|----------------|
| **Grammar Inducer** | Learns token distributions from ASTs | `induce_grammar()` - Bayesian update every 5 gens |
| **ARC Data Loader** | Loads official Kaggle JSON tasks | `load_arc_task()` - Dynamic file discovery |
| **Surrogate Model** | Pre-filters candidates before evaluation | Generate 2×pop, select top 50% by predicted fitness |
| **TDR System** | Atomic patching with rollback | `apply_patch_safe()` - Syntax + import validation |
| **MAP-Elites** | Diversity preservation | Behavioral characterization grid |

### Evolution Pipeline

```
[Random Init] → [EDA Grammar Learning] → [Guided Generation] 
     ↓                    ↓                       ↓
[Surrogate Filter] → [Evaluation] → [AST Analysis] → [Update Grammar Weights]
```

---

## 🧪 Experimental Setup

### Benchmark Tasks

1. **Algorithmic** (`sort`, `reverse`, `max`, `filter`)
   - List manipulation with length 3-12
   - Holdout generalization to length 15-20

2. **ARC (Abstract Reasoning Corpus)**
   - Real JSON files from Kaggle competition
   - Grid transformations (rotation, inversion, scaling)
   - Format: `ARC_GYM/*.json`

3. **Regression** (legacy)
   - `poly2`, `poly3`, `sinmix`, `absline`

### Fitness Function

```python
# Composite Score
score = holdout_error + λ * complexity
where λ = 0.0001 (complexity penalty)
```

---

## 📂 File Structure

```
RSI_REPO/
├── L2_UNIFIED_RSI.py          # Core engine (1600+ lines)
├── test_suite.py              # Comprehensive verification (4 tests)
├── ARC_GYM/
│   ├── 25d8a9c8.json         # Real ARC task (Kaggle format)
│   └── README.md             # Dataset documentation
└── .rsi_state/               # Runtime state persistence
```

---

## 🚀 Usage

### Run Test Suite
```bash
python test_suite.py
```

### Single Benchmark
```python
from L2_UNIFIED_RSI import TaskSpec, Universe, MetaState, FunctionLibrary
import time

task = TaskSpec(name='arc_25d8a9c8', x_min=3, x_max=3)
meta = MetaState()
uni = Universe(uid=1, seed=int(time.time()), meta=meta, pool=[], library=FunctionLibrary())

for gen in range(20):
    uni.step(gen, task, pop_size=100)
    print(f"Gen {gen}: Best={uni.best_score:.4f}")
```

---

## 🔬 Technical Details

### AST-Guided Grammar Induction

**Algorithm**:
1. Select top 20% of population (elites)
2. Parse each genome into Abstract Syntax Tree
3. Count node types: `Call`, `BinOp`, `Name`, `Constant`
4. Update global grammar probabilities via exponential moving average:
   ```python
   GRAMMAR_PROBS[k] = 0.8 * old + 0.2 * (counts[k] / total) * 100
   ```
5. Use weighted sampling in `_random_expr()` for next generation

**Theoretical Basis**: Estimation of Distribution Algorithms (EDA) - learns probabilistic model of search space from successful samples.

### Safety Mechanisms

1. **Step Limit Injection**: AST transformer adds `_steps` counter to every loop/statement
2. **Sandboxed Execution**: Restricted namespace (`SAFE_FUNCS`, `SAFE_BUILTINS`)
3. **Code Validation**: Whitelist-based AST visitor rejects dangerous constructs
4. **Atomic Patching**: Backup → Apply → Verify → Commit/Rollback

---

## 📄 License

MIT License
