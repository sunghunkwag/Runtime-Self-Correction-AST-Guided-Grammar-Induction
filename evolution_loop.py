import subprocess
import sys
import time
import os

def run_evolution():
    print("Starting Evolution Loop (100 Iterations)...", flush=True)

    # 1. Base Evolution (Iterations 1-20): Standard Tasks, Low Difficulty
    # Skipped as per instructions
    # for i in range(1, 21):
    #     print(f"\n--- Iteration {i}/100 [Base Phase] ---", flush=True)
    #     cmd = [
    #         sys.executable, "UNIFIED_RSI_EXTENDED.py", "evolve",
    #         "--generations", "5",
    #         "--population", "32",
    #         "--task", "sort",
    #         "--mode", "solver"
    #     ]
    #
    #     # Iteration 1 starts fresh, subsequent resume
    #     if i == 1:
    #         cmd.append("--fresh")
    #     else:
    #         cmd.append("--resume")
    #
    #     ret = subprocess.run(cmd)
    #     if ret.returncode != 0:
    #         print(f"Iteration {i} failed!", flush=True)
    #         # break? or continue?

    # 2. Advanced Phase (Iterations 21-60): Harder Tasks, Learner Mode (TTT)
    # Skipped as per instructions
    # for i in range(21, 61):
    #     print(f"\n--- Iteration {i}/100 [Advanced Phase - TTT] ---", flush=True)
    #     cmd = [
    #         sys.executable, "UNIFIED_RSI_EXTENDED.py", "learner-evolve",
    #         "--generations", "5",
    #         "--population", "32",
    #         "--task", "reverse",
    #         "--freeze-eval",
    #         "--resume"
    #     ]
    #
    #     ret = subprocess.run(cmd)
    #     if ret.returncode != 0:
    #         print(f"Iteration {i} failed!", flush=True)

    # 3. Expert Phase (Iterations 61-100): Strict Validation, Meta-Meta
    for i in range(61, 101):
        print(f"\n--- Iteration {i}/100 [Expert Phase - Discovery] ---", flush=True)
        cmd = [
            sys.executable, "UNIFIED_RSI_EXTENDED.py", "meta-meta",
            "--episodes", "1",
            "--gens-per-episode", "5",
            "--population", "32",
            "--universes", "2"
        ]

        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            print(f"Iteration {i} failed!", flush=True)

    print("\nEvolution Loop Complete.", flush=True)

if __name__ == "__main__":
    run_evolution()
