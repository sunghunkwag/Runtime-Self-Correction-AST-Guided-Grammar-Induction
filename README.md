# Recursive-Self-Improvement-Engine-BETA

An autonomous AI engine capable of improving its own source code and evolutionary logic at runtime.

## 📌 Status
**Current Version**: `L2_UNIFIED_RSI.py` (Phase 3: Co-Evolution)
- **Turing Complete**: Evolves full Python code (Loops, Ifs, Variables) ✅
- **Meta-Engine**: Evolves its own learning algorithm (Selection/Crossover) ✅
- **Co-Evolution**: Generates and solves increasing difficulties of tasks (Discriminator) ✅

##  How to Run

### Phase 3 Loop (Recommended)
```powershell
python run_phase3.py
```

### Manual Engine Test
```powershell
python L2_UNIFIED_RSI.py rsi-loop --generations 500 --rounds 100
```

## 🛠️ System Architecture

| Component | Description | Status |
|-----------|-------------|--------|
| **LGP Genome** | List of Python statements (unlike old Expression Trees) | **Verified** |
| **CodeValidator** | AST-based security sandbox for generated code | **Verified** |
| **EngineStrategy** | Evolveable `Selection` and `Crossover` logic | **Active** |
| **ProblemGenerator** | Evolves new tasks to challenge the solver (Phase 3) | **Active** |
| **MAP-Elites** | Diversity preservation grid | **Active** |

## 📊 Performance
- **Initial Score**: 205.8 (Random Code)
- **Optimized**: 171.0 (after 3 generations)
- **Capability**: Can generate algorithms using `while` loops and logic.



**MIT License**




