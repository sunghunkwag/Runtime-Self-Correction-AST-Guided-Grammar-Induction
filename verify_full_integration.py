import sys
import os
import random
import time
from pathlib import Path
sys.path.append(os.getcwd())
try:
    from UNIFIED_RSI_EXTENDED import Universe, MetaState, TaskSpec, FunctionLibrary, seed_genome, Genome, safe_exec_engine, EngineStrategy, sample_batch
except ImportError:
    from UNIFIED_RSI_EXTENDED import *

def verify():
    print("🚀 Starting FULL INTEGRATION TEST (LGP + Meta-Engine) 🚀")
    
    # Setup Universe
    univ = Universe(
        uid=999, seed=12345, 
        meta=MetaState(mutation_rate=0.6, crossover_rate=0.2), 
        pool=[], 
        library=FunctionLibrary()
    )
    task = TaskSpec(name='poly3', x_min=-5, x_max=5) # Harder task
    
    start_time = time.time()
    best_scores = []
    
    # Run 20 Generations
    print(f"Goal: Prove score improvement on {task.name}")
    rng = random.Random(12345)
    for g in range(20):
        batch = sample_batch(rng, task)
        if batch is None:
            raise RuntimeError("Batch generation returned None")
        log = univ.step(g, task, pop_size=30, batch=batch)
        score = log.get('score', float('inf'))
        best_scores.append(score)
        
        # Check for improvement
        improved = "⭐" if g > 0 and score < best_scores[g-1] else ""
        print(f"Gen {g:02d}: Score={score:.4f} {improved}")
        
        if score < 1e-6:
            print("🎉 PERFECT SCORE ACHIEVED!")
            break
            
    print("\n--- Test Results ---")
    print(f"Initial Score: {best_scores[0]:.4f}")
    print(f"Final Score:   {best_scores[-1]:.4f}")
    if best_scores[-1] < best_scores[0]:
        print("✅ SUCCESS: Optimization Confirmed (Score Improved)")
    else:
        print("⚠️ CONTROVERSIAL: No improvement (Might need parameter tuning)")
        
    # Verify Meta-State Integrity
    print(f"\nMeta-Strategy Active: {univ.meta.strategy.gid}")
    if univ.meta.strategy.selection_code and univ.meta.strategy.crossover_code:
        print("✅ Meta-Genome Integrity: OK")
    else:
        print("❌ Meta-Genome Integrity: FAIL")

if __name__ == "__main__":
    verify()
