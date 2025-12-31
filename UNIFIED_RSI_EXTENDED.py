"""
UNIFIED_RSI_EXTENDED.py

True RSI (Recursive Self-Improvement) Engine - BETA (Executable)
================================================================

CLI:
  python UNIFIED_RSI_EXTENDED.py selftest
  python UNIFIED_RSI_EXTENDED.py evolve --fresh --generations 100
  python UNIFIED_RSI_EXTENDED.py evolve --fresh --generations 50 --mode program
  python UNIFIED_RSI_EXTENDED.py evolve --fresh --generations 50 --mode algo --task sort_int_list
  python UNIFIED_RSI_EXTENDED.py learner-evolve --fresh --generations 100
  python UNIFIED_RSI_EXTENDED.py meta-meta --episodes 20 --gens-per-episode 20
  python UNIFIED_RSI_EXTENDED.py task-switch --task-a poly2 --task-b piecewise
  python UNIFIED_RSI_EXTENDED.py report --state-dir .rsi_state
  python UNIFIED_RSI_EXTENDED.py transfer-bench --from poly2 --to piecewise --budget 10
  python UNIFIED_RSI_EXTENDED.py rsi-loop --generations 50 --rounds 10
  python UNIFIED_RSI_EXTENDED.py rsi-loop --generations 20 --rounds 5 --mode learner
  python UNIFIED_RSI_EXTENDED.py duo-loop --rounds 5 --slice-seconds 8 --blackboard .rsi_blackboard.jsonl --k-full 6

DUO-LOOP OVERVIEW
-----------------
The duo-loop command adds a sequential, low-spec cooperative loop with two virtual agents:
- Creator: proposes diverse candidate programs (novelty-biased generation).
- Critic: prefilters, refines, stress-checks, and adopts candidates (robustness/generalization-biased).

Unlike evolve/rsi-loop, duo-loop never adopts directly from Creator; adoption is Critic-only.
It keeps state in the existing state directory and logs to an append-only blackboard JSONL file.

CHANGELOG
---------
L0: Solver supports expression genomes and strict program-mode genomes (Assign/If/Return only).
L1: RuleDSL controls mutation/crossover/novelty/acceptance/curriculum knobs per generation.
Metrics: frozen train/hold/stress/test sets, per-gen logs, and transfer report (AUC/regret/recovery/gap).
Algo: Added algorithmic task suite, algo-mode validator/sandbox, and transfer-bench command.
"""
from __future__ import annotations

import argparse
import ast
import collections
import difflib
import hashlib
import json
import math
import copy
import os
import random
import re
import subprocess
import shutil
import sys
import tempfile
import textwrap
import time
import traceback
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Set, Union
import multiprocessing as mp


# ---------------------------
# Utilities
# ---------------------------

def now_ms() -> int:
    return int(time.time() * 1000)

def sha256(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def read_json(p: Path) -> Dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

def write_json(p: Path, obj: Any, indent: int = 2):
    safe_mkdir(p.parent)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=indent, default=str), encoding="utf-8")

def unified_diff(old: str, new: str, name: str) -> str:
    return "".join(
        difflib.unified_diff(
            old.splitlines(True),
            new.splitlines(True),
            fromfile=name,
            tofile=name,
        )
    )


class RunLogger:
    def __init__(self, path: Path, window: int = 10, append: bool = False):
        self.path = path
        self.window = window
        self.records: List[Dict[str, Any]] = []
        self.best_scores: List[float] = []
        self.best_hold: List[float] = []
        self.seen_hashes: Set[str] = set()
        safe_mkdir(self.path.parent)
        if self.path.exists() and not append:
            self.path.unlink()

    def _window_slice(self, vals: List[float]) -> List[float]:
        if not vals:
            return []
        return vals[-self.window :]

    def log(
        self,
        gen: int,
        task_id: str,
        mode: str,
        score_hold: float,
        score_stress: float,
        score_test: float,
        runtime_ms: int,
        nodes: int,
        code_hash: str,
        accepted: bool,
        novelty: float,
        meta_policy_params: Dict[str, Any],
        solver_hash: Optional[str] = None,
        p1_hash: Optional[str] = None,
        err_hold: Optional[float] = None,
        err_stress: Optional[float] = None,
        err_test: Optional[float] = None,
        steps: Optional[int] = None,
        timeout_rate: Optional[float] = None,
        counterexample_count: Optional[int] = None,
        library_size: Optional[int] = None,
        control_packet: Optional[Dict[str, Any]] = None,
        task_descriptor: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.best_scores.append(score_hold)
        self.best_hold.append(score_hold)
        window_vals = self._window_slice(self.best_hold)
        auc_window = sum(window_vals) / max(1, len(window_vals))
        if len(self.best_hold) > self.window:
            delta_best_window = self.best_hold[-1] - self.best_hold[-self.window]
        else:
            delta_best_window = self.best_hold[-1] - self.best_hold[0]
        record = {
            "gen": gen,
            "task_id": task_id,
            "solver_hash": solver_hash or code_hash,
            "p1_hash": p1_hash or "default",
            "mode": mode,
            "score_hold": score_hold,
            "score_stress": score_stress,
            "score_test": score_test,
            "err_hold": err_hold if err_hold is not None else score_hold,
            "err_stress": err_stress if err_stress is not None else score_stress,
            "err_test": err_test if err_test is not None else score_test,
            "auc_window": auc_window,
            "delta_best_window": delta_best_window,
            "runtime_ms": runtime_ms,
            "nodes": nodes,
            "hash": code_hash,
            "accepted": accepted,
            "novelty": novelty,
            "meta_policy_params": meta_policy_params,
            "steps": steps,
            "timeout_rate": timeout_rate,
            "counterexample_count": counterexample_count,
            "library_size": library_size,
            "control_packet": control_packet or {},
            "task_descriptor": task_descriptor,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        self.records.append(record)
        return record


# ---------------------------
# Blackboard utilities
# ---------------------------

def append_blackboard(path: Path, record: Dict[str, Any]) -> None:
    safe_mkdir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def tail_blackboard(path: Path, k: int) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    lines: collections.deque[str] = collections.deque(maxlen=k)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                lines.append(line)
    records = []
    for line in lines:
        try:
            records.append(json.loads(line))
        except Exception:
            continue
    return records


# ---------------------------
# Invention Engine (RSI Integration)
# ---------------------------

@dataclass
class InventionProgramCandidate:
    candidate_id: str
    code: str
    origin: str
    parent_id: Optional[str] = None
    score: float = 0.011744
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    features: Dict[str, int] = field(default_factory=dict)


class InventionRepresentation:
    """Expandable grammar and primitives.

    This enables invention by allowing new control patterns to be introduced
    dynamically, rather than committing to a fixed syntax whitelist.
    """

    def __init__(self) -> None:
        self.grammar: Dict[str, List[Callable[["InventionRepresentation"], str]]] = {
            "program": [self._base_program],
            "solver": [self._solver_template],
            "control": [self._loop_control, self._recursion_control],
            "strategy": [
                self._greedy_strategy,
                self._dp_strategy,
                self._divide_conquer_strategy,
                self._search_strategy,
            ],
        }
        self.library: List[str] = []

    def add_production(self, symbol: str, producer: Callable[["InventionRepresentation"], str]) -> None:
        self.grammar.setdefault(symbol, []).append(producer)

    def expand(self, symbol: str) -> str:
        options = self.grammar.get(symbol, [])
        if not options:
            raise ValueError(f"No productions for symbol: {symbol}")
        return random.choice(options)(self)

    def _base_program(self, _: "InventionRepresentation") -> str:
        helpers = "\n\n".join(self.library) if self.library else ""
        solver = self.expand("solver")
        return textwrap.dedent(
            f"""
            {helpers}

            {solver}
            """
        ).strip()

    def _solver_template(self, _: "InventionRepresentation") -> str:
        control = self.expand("control")
        strategy = self.expand("strategy")
        return textwrap.dedent(
            f"""
            def solve(task):
                \"\"\"Return the solution for the provided task.

                Generated as a full Python function so new control flow patterns
                can be invented, replaced, or expanded.
                \"\"\"
                {control}
                {strategy}
            """
        ).strip()

    def _loop_control(self, _: "InventionRepresentation") -> str:
        return textwrap.indent(
            textwrap.dedent(
                """
                for attempt in range(3):
                    if getattr(task, 'hint', None):
                        break
                """
            ).strip(),
            "    ",
        )

    def _recursion_control(self, _: "InventionRepresentation") -> str:
        return textwrap.indent(
            textwrap.dedent(
                """
                def recur(state, depth):
                    if depth <= 0:
                        return state
                    return recur(state, depth - 1)
                recur(None, 1)
                """
            ).strip(),
            "    ",
        )

    def _greedy_strategy(self, _: "InventionRepresentation") -> str:
        return textwrap.indent(
            textwrap.dedent(
                """
                if task.kind == 'sequence':
                    return [x + 1 for x in task.input]
                if task.kind == 'path':
                    return task.heuristic_path()
                if task.kind == 'transform':
                    return ''.join(sorted(task.input))
                if task.kind == 'aggregate':
                    if getattr(task, 'hint', None) == 'max':
                        return max(task.input)
                    if getattr(task, 'hint', None) == 'min':
                        return min(task.input)
                    if getattr(task, 'hint', None) == 'len':
                        return len(task.input)
                    return sum(task.input)
                return task.fallback()
                """
            ).strip(),
            "    ",
        )

    def _dp_strategy(self, _: "InventionRepresentation") -> str:
        return textwrap.indent(
            textwrap.dedent(
                """
                if task.kind == 'sequence':
                    dp = {0: task.input[0] if task.input else 0}
                    for i in range(1, len(task.input)):
                        dp[i] = dp[i - 1] + task.input[i]
                    return [dp[i] for i in range(len(task.input))]
                if task.kind == 'path':
                    return task.shortest_path()
                if task.kind == 'transform':
                    memo = {}
                    def best(s):
                        if s in memo:
                            return memo[s]
                        if not s:
                            return ''
                        memo[s] = min(s[0] + best(s[1:]), ''.join(sorted(s)))
                        return memo[s]
                    return best(task.input)
                if task.kind == 'aggregate':
                    return sum(task.input)
                return task.fallback()
                """
            ).strip(),
            "    ",
        )

    def _divide_conquer_strategy(self, _: "InventionRepresentation") -> str:
        return textwrap.indent(
            textwrap.dedent(
                """
                if task.kind == 'sequence':
                    def combine(arr):
                        if len(arr) <= 1:
                            return arr
                        mid = len(arr) // 2
                        left = combine(arr[:mid])
                        right = combine(arr[mid:])
                        return [sum(left)] + [sum(right)]
                    return combine(task.input)
                if task.kind == 'path':
                    return task.path_via_split()
                if task.kind == 'transform':
                    def merge_sort(s):
                        if len(s) <= 1:
                            return s
                        mid = len(s) // 2
                        left = merge_sort(s[:mid])
                        right = merge_sort(s[mid:])
                        result = ''
                        while left and right:
                            if left[0] < right[0]:
                                result += left[0]
                                left = left[1:]
                            else:
                                result += right[0]
                                right = right[1:]
                        return result + left + right
                    return merge_sort(task.input)
                return task.fallback()
                """
            ).strip(),
            "    ",
        )

    def _search_strategy(self, _: "InventionRepresentation") -> str:
        return textwrap.indent(
            textwrap.dedent(
                """
                if task.kind == 'sequence':
                    best = None
                    for offset in range(1, 4):
                        candidate = [x + offset for x in task.input]
                        if best is None or sum(candidate) < sum(best):
                            best = candidate
                    return best
                if task.kind == 'path':
                    return task.search()
                if task.kind == 'transform':
                    best = min(task.input, ''.join(sorted(task.input)))
                    return best
                return task.fallback()
                """
            ).strip(),
            "    ",
        )


class InventionProgramGenerator:
    """Generate programs via grammar and composition.

    Composition across a growing library enables reuse of learned abstractions.
    """

    def __init__(self, representation: InventionRepresentation) -> None:
        self.representation = representation
        self.operator_weights: Dict[str, float] = {
            "grammar": 1.0,
            "compose": 1.0,
        }

    def generate(self) -> InventionProgramCandidate:
        operator = self._choose_operator()
        if operator == "compose" and self.representation.library:
            return self._compose_program()
        return self._grammar_program()

    def _choose_operator(self) -> str:
        total = sum(self.operator_weights.values())
        roll = random.random() * total
        cumulative = 0.0
        for name, weight in self.operator_weights.items():
            cumulative += weight
            if roll <= cumulative:
                return name
        return "grammar"

    def _grammar_program(self) -> InventionProgramCandidate:
        code = self.representation.expand("program")
        return InventionProgramCandidate(candidate_id=sha256(code + str(time.time())), code=code, origin="grammar")

    def _compose_program(self) -> InventionProgramCandidate:
        helpers = random.sample(self.representation.library, k=1)
        base = self.representation.expand("program")
        code = "\n\n".join(helpers + [base])
        return InventionProgramCandidate(candidate_id=sha256(code + str(time.time())), code=code, origin="compose")


@dataclass
class InventionTask:
    kind: str
    input: Any
    expected: Any
    hint: Optional[str] = None
    descriptor: Dict[str, Any] = field(default_factory=dict)

    def heuristic_path(self) -> Any:
        return self.expected

    def shortest_path(self) -> Any:
        return self.expected

    def path_via_split(self) -> Any:
        return self.expected

    def search(self) -> Any:
        return self.expected

    def fallback(self) -> Any:
        return self.expected


class ProblemGenerator:
    """Mutates and creates tasks continuously to avoid a fixed finite set."""

    def __init__(self) -> None:
        self.seed = 0
        self.base_kinds = ["sequence", "path", "transform", "aggregate"]
        self.transform_ops = ["sort", "reverse", "unique", "shift"]
        self.aggregate_ops = ["sum", "max", "min", "len"]

    def generate_tasks(
        self,
        count: int = 3,
        parents: Optional[List[InventionTask]] = None,
    ) -> List[InventionTask]:
        tasks: List[InventionTask] = []
        for _ in range(count):
            self.seed += 1
            random.seed(self.seed + random.randint(0, 9999))
            if parents and random.random() < 0.5:
                parent = random.choice(parents)
                tasks.append(self.mutate_task(parent))
            else:
                tasks.append(self.create_task())
        return tasks

    def create_task(self) -> InventionTask:
        kind = random.choice(self.base_kinds + [f"transform:{random.choice(self.transform_ops)}"])
        if kind == "sequence":
            data = [random.randint(1, 7) for _ in range(random.randint(3, 6))]
            expected = [sum(data[:i + 1]) for i in range(len(data))]
            return InventionTask(kind=kind, input=data, expected=expected, hint="prefix")
        if kind == "path":
            size = random.randint(3, 5)
            grid = [[random.randint(1, 9) for _ in range(size)] for _ in range(size)]
            expected = sum(grid[0]) + sum(row[-1] for row in grid[1:])
            return InventionTask(kind=kind, input=grid, expected=expected, hint="grid")
        if kind.startswith("transform"):
            op = kind.split(":", 1)[1] if ":" in kind else random.choice(self.transform_ops)
            word = "".join(random.choice("abcde") for _ in range(random.randint(4, 7)))
            expected = self._apply_transform(op, word)
            return InventionTask(kind="transform", input=word, expected=expected, hint=op, descriptor={"op": op})
        op = random.choice(self.aggregate_ops)
        data = [random.randint(1, 9) for _ in range(random.randint(3, 6))]
        expected = self._apply_aggregate(op, data)
        return InventionTask(kind="aggregate", input=data, expected=expected, hint=op, descriptor={"op": op})

    def mutate_task(self, task: InventionTask) -> InventionTask:
        if task.kind == "sequence":
            data = [x + random.choice([-1, 0, 1]) for x in task.input]
            data.append(random.randint(1, 7))
            expected = [sum(data[:i + 1]) for i in range(len(data))]
            return InventionTask(kind=task.kind, input=data, expected=expected, hint=task.hint, descriptor=task.descriptor)
        if task.kind == "path":
            grid = [row[:] for row in task.input]
            r = random.randint(0, len(grid) - 1)
            c = random.randint(0, len(grid[0]) - 1)
            grid[r][c] = max(1, grid[r][c] + random.choice([-2, -1, 1, 2]))
            expected = sum(grid[0]) + sum(row[-1] for row in grid[1:])
            return InventionTask(kind=task.kind, input=grid, expected=expected, hint=task.hint, descriptor=task.descriptor)
        if task.kind == "transform":
            op = task.descriptor.get("op", random.choice(self.transform_ops))
            word = task.input + random.choice("abcde")
            expected = self._apply_transform(op, word)
            return InventionTask(kind="transform", input=word, expected=expected, hint=op, descriptor={"op": op})
        if task.kind == "aggregate":
            op = task.descriptor.get("op", random.choice(self.aggregate_ops))
            data = task.input + [random.randint(1, 9)]
            expected = self._apply_aggregate(op, data)
            return InventionTask(kind="aggregate", input=data, expected=expected, hint=op, descriptor={"op": op})
        return self.create_task()

    def _apply_transform(self, op: str, word: str) -> str:
        if op == "sort":
            return "".join(sorted(word))
        if op == "reverse":
            return word[::-1]
        if op == "unique":
            return "".join(dict.fromkeys(word))
        if op == "shift":
            return "".join(chr(((ord(ch) - 97 + 1) % 26) + 97) for ch in word)
        return word

    def _apply_aggregate(self, op: str, data: List[int]) -> Any:
        if op == "sum":
            return sum(data)
        if op == "max":
            return max(data)
        if op == "min":
            return min(data)
        if op == "len":
            return len(data)
        return sum(data)


@dataclass
class RewardModel:
    performance_weight: float = 1.0
    transfer_weight: float = 0.7
    reuse_weight: float = 0.480815
    compression_weight: float = 0.3

    def score(self, metrics: Dict[str, float]) -> float:
        return (
            self.performance_weight * metrics.get("performance", 0.0)
            + self.transfer_weight * metrics.get("transfer", 0.0)
            + self.reuse_weight * metrics.get("reuse", 0.0)
            + self.compression_weight * metrics.get("compression", 0.0)
        )


@dataclass
class CandidateRecord:
    candidate_id: str
    parent_id: Optional[str]
    origin: str
    code: str
    score: float
    metrics: Dict[str, float]
    timestamp_ms: int


class InventionArchive:
    """Archive with lineage and a reusable subroutine pool."""

    def __init__(self, promotion_threshold: int = 2) -> None:
        self.records: List[CandidateRecord] = []
        self.lineage: Dict[str, CandidateRecord] = {}
        self.subroutine_pool: Dict[str, int] = {}
        self.promotion_threshold = promotion_threshold

    def add(self, candidate: InventionProgramCandidate) -> None:
        metrics = candidate.diagnostics.get("metrics", {})
        record = CandidateRecord(
            candidate_id=candidate.candidate_id,
            parent_id=candidate.parent_id,
            origin=candidate.origin,
            code=candidate.code,
            score=candidate.score,
            metrics=metrics,
            timestamp_ms=now_ms(),
        )
        self.records.append(record)
        self.lineage[candidate.candidate_id] = record

    def note_subroutine(self, snippet: str) -> bool:
        count = self.subroutine_pool.get(snippet, 0) + 1
        self.subroutine_pool[snippet] = count
        return count >= self.promotion_threshold


class Searcher:
    name: str = "base"

    def propose(
        self,
        representation: InventionRepresentation,
        archive: InventionArchive,
        problem_generator: ProblemGenerator,
    ) -> InventionProgramCandidate:
        raise NotImplementedError


class LocalEditSearcher(Searcher):
    name = "local_edit"

    def propose(
        self,
        representation: InventionRepresentation,
        archive: InventionArchive,
        problem_generator: ProblemGenerator,
    ) -> InventionProgramCandidate:
        source = representation.expand("program")
        if archive.records:
            source = random.choice(archive.records).code
        mutated = self._mutate_code(source)
        return InventionProgramCandidate(candidate_id=sha256(mutated + str(time.time())), code=mutated, origin=self.name)

    def _mutate_code(self, code: str) -> str:
        tree = ast.parse(code)
        constants = [node for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, int)]
        if constants:
            node = random.choice(constants)
            node.value = node.value + random.choice([-1, 1])
            return ast.unparse(tree)
        return code.replace("range(3)", "range(4)", 1)


class StructuralComposeSearcher(Searcher):
    name = "structural_compose"

    def propose(
        self,
        representation: InventionRepresentation,
        archive: InventionArchive,
        problem_generator: ProblemGenerator,
    ) -> InventionProgramCandidate:
        helpers = []
        if representation.library:
            helpers.extend(random.sample(representation.library, k=min(2, len(representation.library))))
        if archive.subroutine_pool:
            helpers.extend(random.sample(list(archive.subroutine_pool.keys()), k=min(1, len(archive.subroutine_pool))))
        base = representation.expand("program")
        code = "\n\n".join(helpers + [base])
        return InventionProgramCandidate(candidate_id=sha256(code + str(time.time())), code=code, origin=self.name)


class RepresentationEditSearcher(Searcher):
    name = "representation_edit"

    def propose(
        self,
        representation: InventionRepresentation,
        archive: InventionArchive,
        problem_generator: ProblemGenerator,
    ) -> InventionProgramCandidate:
        def new_strategy(_: InventionRepresentation) -> str:
            return textwrap.indent(
                textwrap.dedent(
                    """
                    if task.kind == 'aggregate':
                        if getattr(task, 'hint', None) == 'max':
                            return max(task.input)
                        if getattr(task, 'hint', None) == 'min':
                            return min(task.input)
                        return sum(task.input)
                    """
                ).strip(),
                "    ",
            )

        representation.add_production("strategy", new_strategy)
        code = representation.expand("program")
        return InventionProgramCandidate(candidate_id=sha256(code + str(time.time())), code=code, origin=self.name)


class SearcherManager:
    def __init__(self, searchers: List[Searcher]) -> None:
        self.searchers = {s.name: s for s in searchers}
        self.weights: Dict[str, float] = {s.name: 1.0 for s in searchers}

    def propose(
        self,
        representation: InventionRepresentation,
        archive: InventionArchive,
        problem_generator: ProblemGenerator,
    ) -> InventionProgramCandidate:
        searcher = self._select_searcher()
        candidate = self.searchers[searcher].propose(representation, archive, problem_generator)
        candidate.origin = searcher
        return candidate

    def _select_searcher(self) -> str:
        total = sum(self.weights.values())
        roll = random.random() * total
        cumulative = 0.0
        for name, weight in self.weights.items():
            cumulative += weight
            if roll <= cumulative:
                return name
        return next(iter(self.weights))

    def update_weight(self, searcher: str, delta: float) -> None:
        self.weights[searcher] = clamp(self.weights.get(searcher, 1.0) + delta, 0.2, 5.0)


@dataclass
class BudgetLevel:
    name: str
    task_count: int
    transfer_count: int
    survivors: int


class BudgetLadderPolicy:
    """Budget ladder (B1..B4) where only survivors advance."""

    def __init__(self) -> None:
        self.levels = [
            BudgetLevel("B1", task_count=2, transfer_count=1, survivors=4),
            BudgetLevel("B2", task_count=3, transfer_count=2, survivors=3),
            BudgetLevel("B3", task_count=4, transfer_count=3, survivors=2),
            BudgetLevel("B4", task_count=5, transfer_count=4, survivors=1),
        ]

    def run(
        self,
        candidates: List[InventionProgramCandidate],
        problem_generator: ProblemGenerator,
        evaluator: "InventionEvaluator",
        archive: InventionArchive,
        reward_model: RewardModel,
    ) -> List[InventionProgramCandidate]:
        survivors = candidates
        for level in self.levels:
            if not survivors:
                break
            tasks = problem_generator.generate_tasks(level.task_count)
            transfer_tasks = problem_generator.generate_tasks(level.transfer_count, parents=tasks)
            for candidate in survivors:
                evaluator.evaluate(candidate, tasks, transfer_tasks, archive, reward_model)
            survivors = sorted(survivors, key=lambda c: c.score, reverse=True)[: level.survivors]
        return survivors


class InventionEvaluator:
    """Execute candidates in isolated processes and score them.

    Failures become diagnostic signals, enabling the meta-controller to adapt.
    """

    def __init__(self) -> None:
        self.novelty_weight = 0.2
        self.archive_features: List[Dict[str, int]] = []

    def evaluate(
        self,
        candidate: InventionProgramCandidate,
        tasks: List[InventionTask],
        transfer_tasks: List[InventionTask],
        archive: "InventionArchive",
        reward_model: "RewardModel",
        timeout: float = 1.0,
    ) -> None:
        results: List[Tuple[bool, str]] = []
        for task in tasks:
            success, info = self._run_in_subprocess(candidate.code, task, timeout)
            results.append((success, info))
        transfer_results: List[Tuple[bool, str]] = []
        for task in transfer_tasks:
            success, info = self._run_in_subprocess(candidate.code, task, timeout)
            transfer_results.append((success, info))
        candidate.diagnostics["results"] = results
        candidate.diagnostics["transfer_results"] = transfer_results
        candidate.features = self._extract_features(candidate.code)
        metrics = self._score_components(candidate, results, transfer_results, tasks, archive)
        candidate.diagnostics["metrics"] = metrics
        candidate.score = reward_model.score(metrics)
        self.archive_features.append(candidate.features)

    def _run_in_subprocess(self, code: str, task: InventionTask, timeout: float) -> Tuple[bool, str]:
        queue: mp.Queue = mp.Queue()

        def runner() -> None:
            try:
                scope: Dict[str, Any] = {}
                exec(code, scope)
                if "solve" not in scope:
                    queue.put((False, "missing solve"))
                    return
                result = scope["solve"](task)
                queue.put((result == task.expected, repr(result)))
            except Exception:
                queue.put((False, traceback.format_exc()))

        process = mp.Process(target=runner)
        process.start()
        process.join(timeout)
        if process.is_alive():
            process.terminate()
            process.join()
            return False, "timeout"
        if queue.empty():
            return False, "no output"
        return queue.get()

    def _score_components(
        self,
        candidate: InventionProgramCandidate,
        results: List[Tuple[bool, str]],
        transfer_results: List[Tuple[bool, str]],
        tasks: List[InventionTask],
        archive: "InventionArchive",
    ) -> Dict[str, float]:
        success_rate = sum(1 for ok, _ in results if ok) / max(1, len(results))
        transfer_rate = sum(1 for ok, _ in transfer_results if ok) / max(1, len(transfer_results))
        reuse = self._reuse_score(candidate.code, archive)
        compression = self._compression_score(candidate.code)
        novelty = self._novelty(candidate.code)
        anti_trick = -0.2 if self._is_trivial(candidate.code, tasks) else 0.0
        return {
            "performance": success_rate + anti_trick,
            "transfer": transfer_rate,
            "reuse": reuse,
            "compression": compression,
            "novelty": novelty,
        }

    def _novelty(self, code: str) -> float:
        features = self._extract_features(code)
        if not self.archive_features:
            return 1.0
        distances = []
        for past in self.archive_features:
            distance = 0
            for key, value in features.items():
                distance += abs(value - past.get(key, 0))
            distances.append(distance)
        return sum(distances) / len(distances)

    def _is_trivial(self, code: str, tasks: List[InventionTask]) -> bool:
        if "return task.expected" in code:
            return True
        return all(len(repr(task.input)) < 10 for task in tasks) and "for" not in code

    def _extract_features(self, code: str) -> Dict[str, int]:
        tree = ast.parse(code)
        features: Dict[str, int] = {}
        for node in ast.walk(tree):
            name = type(node).__name__
            features[name] = features.get(name, 0) + 1
        return features

    def _reuse_score(self, code: str, archive: "InventionArchive") -> float:
        if not archive.subroutine_pool:
            return 0.0
        hits = 0
        for snippet in archive.subroutine_pool:
            if snippet in code:
                hits += 1
        return hits / max(1, len(archive.subroutine_pool))

    def _compression_score(self, code: str) -> float:
        node_count = sum(1 for _ in ast.walk(ast.parse(code)))
        return 1.0 / (1.0 + node_count / 50.0)


class InventionSelfModifier:
    """Adjusts generator, evaluator, and grammar based on diagnostics.

    This makes the system's learning rules and search operators mutable objects.
    """

    def __init__(
        self,
        representation: InventionRepresentation,
        evaluator: InventionEvaluator,
        searchers: SearcherManager,
        reward_model: RewardModel,
        budget_policy: BudgetLadderPolicy,
    ) -> None:
        self.representation = representation
        self.evaluator = evaluator
        self.searchers = searchers
        self.reward_model = reward_model
        self.budget_policy = budget_policy

    def adapt(self, candidate: InventionProgramCandidate) -> None:
        metrics = candidate.diagnostics.get("metrics", {})
        performance = metrics.get("performance", 0.0)
        transfer = metrics.get("transfer", 0.0)
        reuse = metrics.get("reuse", 0.0)
        if performance < 0.7:
            self.searchers.update_weight("local_edit", 0.2)
            self.evaluator.novelty_weight = min(1.5, self.evaluator.novelty_weight + 0.05)
            self._expand_grammar()
        if transfer < 0.5:
            self.searchers.update_weight("representation_edit", 0.2)
            self.reward_model.transfer_weight = min(1.2, self.reward_model.transfer_weight + 0.1)
        if reuse < 0.2:
            self.searchers.update_weight("structural_compose", 0.2)
            self.reward_model.reuse_weight = min(1.0, self.reward_model.reuse_weight + 0.1)
        if performance > 0.8 and transfer > 0.6:
            for level in self.budget_policy.levels:
                level.task_count = min(level.task_count + 1, 6)

    def _expand_grammar(self) -> None:
        def new_control(_: InventionRepresentation) -> str:
            return textwrap.indent(
                textwrap.dedent(
                    """
                    state = {}
                    if hasattr(task, 'hint'):
                        state['hint'] = task.hint
                    """
                ).strip(),
                "    ",
            )

        self.representation.add_production("control", new_control)


class InventionMetaController:
    """Coordinates generation, evaluation, self-modification, and retention.

    This creates a loop where algorithmic structures can be replaced entirely.
    """

    def __init__(self) -> None:
        self.representation = InventionRepresentation()
        self.evaluator = InventionEvaluator()
        self.problem_generator = ProblemGenerator()
        self.reward_model = RewardModel()
        self.archive = InventionArchive()
        self.searchers = SearcherManager(
            [LocalEditSearcher(), StructuralComposeSearcher(), RepresentationEditSearcher()]
        )
        self.budget_policy = BudgetLadderPolicy()
        self.self_modifier = InventionSelfModifier(
            self.representation,
            self.evaluator,
            self.searchers,
            self.reward_model,
            self.budget_policy,
        )
        self.candidate_history: List[InventionProgramCandidate] = []

    def run(self, iterations: int = 5) -> None:
        for _ in range(iterations):
            candidates = self._generate_candidates(pool_size=8)
            survivors = self.budget_policy.run(
                candidates,
                self.problem_generator,
                self.evaluator,
                self.archive,
                self.reward_model,
            )
            for candidate in survivors:
                self._retain(candidate)
                self.self_modifier.adapt(candidate)

    def _retain(self, candidate: InventionProgramCandidate) -> None:
        if candidate.score <= 0:
            return
        self.archive.add(candidate)
        self.candidate_history.append(candidate)
        self._extract_helpers(candidate.code)
        self._extract_subroutines(candidate.code)

    def _extract_helpers(self, code: str) -> None:
        tree = ast.parse(code)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name != "solve":
                helper_code = ast.unparse(node)
                if helper_code not in self.representation.library:
                    self.representation.library.append(helper_code)

    def _extract_subroutines(self, code: str) -> None:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "solve":
                for child in node.body:
                    snippet = ast.unparse(child)
                    if self.archive.note_subroutine(snippet):
                        self._promote_subroutine(snippet)

    def _promote_subroutine(self, snippet: str) -> None:
        if snippet.strip().startswith("def "):
            if snippet not in self.representation.library:
                self.representation.library.append(snippet)
            return
        name = f"subroutine_{sha256(snippet)[:8]}"
        helper_code = "def " + name + "(task):\n" + textwrap.indent(snippet, "    ") + "\n    return None"
        if helper_code not in self.representation.library:
            self.representation.library.append(helper_code)

    def _generate_candidates(self, pool_size: int) -> List[InventionProgramCandidate]:
        candidates: List[InventionProgramCandidate] = []
        for _ in range(pool_size):
            candidate = self.searchers.propose(self.representation, self.archive, self.problem_generator)
            if self.archive.records:
                candidate.parent_id = random.choice(self.archive.records).candidate_id
            candidates.append(candidate)
        return candidates


def cmd_invention(args):
    random.seed(args.seed)
    mp.set_start_method("spawn", force=True)
    controller = InventionMetaController()
    start = time.time()
    controller.run(iterations=args.iterations)
    duration = time.time() - start
    print(f"Completed {len(controller.archive.records)} retained candidates in {duration:.2f}s")
    return 0


# ---------------------------
# Safe primitives
# ---------------------------

SAFE_FUNCS: Dict[str, Callable] = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "exp": math.exp,
    "tanh": math.tanh,
    "abs": abs,
    "sqrt": lambda x: math.sqrt(abs(x) + 1e-12),
    "log": lambda x: math.log(abs(x) + 1e-12),
    "pow2": lambda x: x * x,
    "sigmoid": lambda x: 1.0 / (1.0 + math.exp(-clamp(x, -500, 500))),
    "gamma": lambda x: math.gamma(abs(x) + 1e-09) if abs(x) < 170 else float("inf"),
    "erf": math.erf,
    "ceil": math.ceil,
    "floor": math.floor,
    "sign": lambda x: math.copysign(1.0, x),
    # list helpers (legacy)
    "sorted": sorted,
    "reversed": reversed,
    "max": max,
    "min": min,
    "sum": sum,
    "len": len,
    "list": list,
}

GRAMMAR_PROBS: Dict[str, float] = {k: 1.0 for k in SAFE_FUNCS}
GRAMMAR_PROBS.update({"binop": 2.0, "call": 15.0, "const": 1.0, "var": 2.0})

SAFE_BUILTINS = {
    "abs": abs,
    "min": min,
    "max": max,
    "float": float,
    "int": int,
    "len": len,
    "range": range,
    "list": list,
    "sorted": sorted,
    "reversed": reversed,
    "sum": sum,
}

# ---------------------------
# Algo-mode safe primitives
# ---------------------------

def make_list(size: int = 0, fill: Any = 0) -> List[Any]:
    size = int(clamp(size, 0, 256))
    return [fill for _ in range(size)]

def list_len(xs: Any) -> int:
    return len(xs) if isinstance(xs, list) else 0

def list_get(xs: Any, idx: int, default: Any = 0) -> Any:
    if not isinstance(xs, list) or not xs:
        return default
    i = int(idx)
    if i < 0:
        i = 0
    if i >= len(xs):
        i = len(xs) - 1
    return xs[i]

def list_set(xs: Any, idx: int, val: Any) -> List[Any]:
    if not isinstance(xs, list):
        return make_list()
    if not xs:
        return [val]
    i = int(idx)
    if i < 0:
        i = 0
    if i >= len(xs):
        i = len(xs) - 1
    ys = list(xs)
    ys[i] = val
    return ys

def list_push(xs: Any, val: Any) -> List[Any]:
    ys = list(xs) if isinstance(xs, list) else []
    if len(ys) >= 256:
        return ys
    ys.append(val)
    return ys

def list_pop(xs: Any, default: Any = 0) -> Tuple[List[Any], Any]:
    ys = list(xs) if isinstance(xs, list) else []
    if not ys:
        return (ys, default)
    val = ys.pop()
    return (ys, val)

def list_swap(xs: Any, i: int, j: int) -> List[Any]:
    ys = list(xs) if isinstance(xs, list) else []
    if not ys:
        return ys
    a = int(clamp(i, 0, len(ys) - 1))
    b = int(clamp(j, 0, len(ys) - 1))
    ys[a], ys[b] = ys[b], ys[a]
    return ys

def list_copy(xs: Any) -> List[Any]:
    return list(xs) if isinstance(xs, list) else []

def make_map() -> Dict[Any, Any]:
    return {}

def map_get(m: Any, key: Any, default: Any = 0) -> Any:
    if not isinstance(m, dict):
        return default
    return m.get(key, default)

def map_set(m: Any, key: Any, val: Any) -> Dict[Any, Any]:
    d = dict(m) if isinstance(m, dict) else {}
    if len(d) >= 256 and key not in d:
        return d
    d[key] = val
    return d

def map_has(m: Any, key: Any) -> bool:
    return isinstance(m, dict) and key in m

def safe_range(n: int, limit: int = 256) -> List[int]:
    n = int(clamp(n, 0, limit))
    return list(range(n))

def safe_irange(a: int, b: int, limit: int = 256) -> List[int]:
    a = int(clamp(a, -limit, limit))
    b = int(clamp(b, -limit, limit))
    if a <= b:
        return list(range(a, b))
    return list(range(a, b, -1))

SAFE_ALGO_FUNCS: Dict[str, Callable] = {
    "make_list": make_list,
    "list_len": list_len,
    "list_get": list_get,
    "list_set": list_set,
    "list_push": list_push,
    "list_pop": list_pop,
    "list_swap": list_swap,
    "list_copy": list_copy,
    "make_map": make_map,
    "map_get": map_get,
    "map_set": map_set,
    "map_has": map_has,
    "safe_range": safe_range,
    "safe_irange": safe_irange,
    "clamp": clamp,
    "abs": abs,
    "min": min,
    "max": max,
    "int": int,
}

SAFE_VARS = {"x"} | {f"v{i}" for i in range(10)}


# grid helpers (ARC-like)
def _g_rot90(g):
    return [list(r) for r in zip(*g[::-1])]

def _g_flip(g):
    return g[::-1]

def _g_inv(g):
    return [[1 - c if c in (0, 1) else c for c in r] for r in g]

def _g_get(g, r, c):
    return g[r % len(g)][c % len(g[0])] if g and g[0] else 0

SAFE_FUNCS.update({"rot90": _g_rot90, "flip": _g_flip, "inv": _g_inv, "get": _g_get})
for k in ["rot90", "flip", "inv", "get"]:
    GRAMMAR_PROBS[k] = 1.0


# ---------------------------
# Safety: step limit + validators
# ---------------------------

class StepLimitExceeded(Exception):
    pass

class StepLimitTransformer(ast.NodeTransformer):
    """Inject step counting into loops and function bodies to prevent non-termination."""

    def __init__(self, limit: int = 5000):
        self.limit = limit

    def _inject_steps(self, node: ast.FunctionDef) -> None:
        glob = ast.Global(names=["_steps"])
        reset = ast.parse("_steps = 0").body[0]
        inc = ast.parse("_steps += 1").body[0]
        check = ast.parse(f"if _steps > {self.limit}: raise StepLimitExceeded()").body[0]
        node.body.insert(0, glob)
        node.body.insert(1, reset)
        node.body.insert(2, inc)
        node.body.insert(3, check)

    def visit_FunctionDef(self, node):
        self._inject_steps(node)
        self.generic_visit(node)
        return node

    def visit_While(self, node):
        inc = ast.parse("_steps += 1").body[0]
        check = ast.parse(f"if _steps > {self.limit}: raise StepLimitExceeded()").body[0]
        node.body.insert(0, inc)
        node.body.insert(1, check)
        self.generic_visit(node)
        return node

    def visit_For(self, node):
        inc = ast.parse("_steps += 1").body[0]
        check = ast.parse(f"if _steps > {self.limit}: raise StepLimitExceeded()").body[0]
        node.body.insert(0, inc)
        node.body.insert(1, check)
        self.generic_visit(node)
        return node


class CodeValidator(ast.NodeVisitor):
    """
    Allow a safe subset of Python: assignments, flow control, simple expressions, calls to safe names.
    Forbid imports, attribute access, comprehensions, lambdas, etc.
    """

    _allowed = [
        ast.Module,
        ast.FunctionDef,
        ast.arguments,
        ast.arg,
        ast.Return,
        ast.Assign,
        ast.AnnAssign,
        ast.AugAssign,
        ast.Name,
        ast.Constant,
        ast.Expr,
        ast.If,
        ast.While,
        ast.For,
        ast.Break,
        ast.Continue,
        ast.BinOp,
        ast.UnaryOp,
        ast.Compare,
        ast.Call,
        ast.List,
        ast.Tuple,  # critical for tuple-assign (swap)
        ast.Dict,
        ast.Set,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
        ast.Attribute,
        ast.Subscript,
        ast.Slice,
        ast.Load,
        ast.Store,
        ast.IfExp,
        ast.operator,
        ast.boolop,
        ast.unaryop,
        ast.cmpop,
    ]
    if hasattr(ast, "Index"):
        _allowed.append(ast.Index)

    ALLOWED = tuple(_allowed)

    def __init__(self):
        self.ok, self.err = (True, None)

    def visit(self, node):
        if not isinstance(node, self.ALLOWED):
            self.ok, self.err = (False, f"Forbidden: {type(node).__name__}")
            return
        if isinstance(node, ast.Name):
            if node.id.startswith("__") or node.id in ("open", "eval", "exec", "compile", "__import__", "globals", "locals"):
                self.ok, self.err = (False, f"Forbidden name: {node.id}")
                return
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                self.ok, self.err = (False, f"Forbidden attribute: {node.attr}")
                return
        if isinstance(node, ast.Call):
            # forbid attribute calls (e.g., os.system)
            if not isinstance(node.func, (ast.Name, ast.Attribute)):
                self.ok, self.err = (False, "Forbidden call form (non-Name/Attribute callee)")
                return
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name) and node.value.id in SAFE_BUILTINS:
                self.ok, self.err = (False, "Forbidden subscript on builtin")
                return
        super().generic_visit(node)

def validate_code(code: str) -> Tuple[bool, str]:
    try:
        tree = ast.parse(code)
        v = CodeValidator()
        v.visit(tree)
        return (v.ok, v.err or "")
    except Exception as e:
        return (False, str(e))


class ProgramValidator(ast.NodeVisitor):
    """Strict program-mode validator: Assign/If/Return only, no loops or attributes."""

    _allowed = [
        ast.Module,
        ast.FunctionDef,
        ast.arguments,
        ast.arg,
        ast.Return,
        ast.Assign,
        ast.Name,
        ast.Constant,
        ast.Expr,
        ast.If,
        ast.BinOp,
        ast.UnaryOp,
        ast.Compare,
        ast.Call,
        ast.List,
        ast.Tuple,
        ast.Dict,
        ast.Set,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
        ast.Attribute,
        ast.Subscript,
        ast.Slice,
        ast.Load,
        ast.Store,
        ast.IfExp,
        ast.operator,
        ast.boolop,
        ast.unaryop,
        ast.cmpop,
    ]
    if hasattr(ast, "Index"):
        _allowed.append(ast.Index)

    ALLOWED = tuple(_allowed)

    def __init__(self):
        self.ok, self.err = (True, None)

    def visit(self, node):
        if not isinstance(node, self.ALLOWED):
            self.ok, self.err = (False, f"Forbidden program node: {type(node).__name__}")
            return
        if isinstance(node, ast.Name):
            if node.id.startswith("__") or node.id in ("open", "eval", "exec", "compile", "__import__", "globals", "locals"):
                self.ok, self.err = (False, f"Forbidden name: {node.id}")
                return
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                self.ok, self.err = (False, f"Forbidden attribute: {node.attr}")
                return
        if isinstance(node, ast.Call):
            if not isinstance(node.func, (ast.Name, ast.Attribute)):
                self.ok, self.err = (False, "Forbidden call form (non-Name/Attribute callee)")
                return
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name) and node.value.id in SAFE_BUILTINS:
                self.ok, self.err = (False, "Forbidden subscript on builtin")
                return
        super().generic_visit(node)


def validate_program(code: str) -> Tuple[bool, str]:
    try:
        tree = ast.parse(code)
        v = ProgramValidator()
        v.visit(tree)
        return (v.ok, v.err or "")
    except Exception as e:
        return (False, str(e))


class AlgoProgramValidator(ast.NodeVisitor):
    """Algo-mode validator with bounded structure and constrained attribute access."""

    _allowed = [
        ast.Module,
        ast.FunctionDef,
        ast.arguments,
        ast.arg,
        ast.Return,
        ast.Assign,
        ast.Name,
        ast.Constant,
        ast.Expr,
        ast.If,
        ast.For,
        ast.While,
        ast.BinOp,
        ast.UnaryOp,
        ast.Compare,
        ast.BoolOp,
        ast.IfExp,
        ast.Call,
        ast.List,
        ast.Tuple,
        ast.Dict,
        ast.Set,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
        ast.Attribute,
        ast.Subscript,
        ast.Load,
        ast.Store,
        ast.operator,
        ast.boolop,
        ast.unaryop,
        ast.cmpop,
    ]
    if hasattr(ast, "Index"):
        _allowed.append(ast.Index)

    ALLOWED = tuple(_allowed)

    def __init__(self):
        self.ok, self.err = (True, None)

    def visit(self, node):
        if not isinstance(node, self.ALLOWED):
            self.ok, self.err = (False, f"Forbidden: {type(node).__name__}")
            return
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                self.ok, self.err = (False, f"Forbidden attribute: {node.attr}")
                return
        if isinstance(node, ast.Name):
            if node.id.startswith("__") or node.id in ("open", "eval", "exec", "compile", "__import__", "globals", "locals"):
                self.ok, self.err = (False, f"Forbidden name: {node.id}")
                return
        if isinstance(node, ast.Call):
            if not isinstance(node.func, (ast.Name, ast.Attribute)):
                self.ok, self.err = (False, "Forbidden call form (non-Name/Attribute callee)")
                return
        super().generic_visit(node)


def algo_program_limits_ok(
    code: str,
    max_nodes: int = 420,
    max_depth: int = 32,
    max_funcs: int = 8,
    max_locals: int = 48,
    max_consts: int = 128,
    max_subscripts: int = 64,
) -> bool:
    try:
        tree = ast.parse(code)
    except Exception:
        return False
    nodes = sum(1 for _ in ast.walk(tree))
    depth = ast_depth(code)
    funcs = sum(1 for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
    locals_set = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    consts = sum(1 for n in ast.walk(tree) if isinstance(n, ast.Constant))
    subs = sum(1 for n in ast.walk(tree) if isinstance(n, ast.Subscript))
    return (
        nodes <= max_nodes
        and depth <= max_depth
        and funcs <= max_funcs
        and len(locals_set) <= max_locals
        and consts <= max_consts
        and subs <= max_subscripts
    )


def validate_algo_program(code: str) -> Tuple[bool, str]:
    try:
        tree = ast.parse(code)
        v = AlgoProgramValidator()
        v.visit(tree)
        if not v.ok:
            return (False, v.err or "")
        if not algo_program_limits_ok(code):
            return (False, "algo_program_limits")
        return (True, "")
    except Exception as e:
        return (False, str(e))


class ExprValidator(ast.NodeVisitor):
    """Validate a single expression (mode='eval') allowing only safe names and safe call forms."""
    ALLOWED = (
        ast.Expression,
        ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp,
        ast.Call,
        ast.Attribute,
        ast.Name, ast.Load,
        ast.Constant,
        ast.List, ast.Tuple, ast.Dict,
        ast.Set,
        ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
        ast.Subscript, ast.Slice,
        ast.operator, ast.unaryop, ast.boolop, ast.cmpop,
    )

    def __init__(self, allowed_names: Set[str]):
        self.allowed_names = allowed_names
        self.ok = True
        self.err: Optional[str] = None

    def visit(self, node):
        if not isinstance(node, self.ALLOWED):
            self.ok, self.err = (False, f"Forbidden expr node: {type(node).__name__}")
            return
        if isinstance(node, ast.Name):
            if node.id.startswith("__") or node.id in ("open", "eval", "exec", "compile", "__import__", "globals", "locals"):
                self.ok, self.err = (False, f"Forbidden name: {node.id}")
                return
            if node.id not in self.allowed_names:
                self.ok, self.err = (False, f"Unknown name: {node.id}")
                return
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                self.ok, self.err = (False, f"Forbidden attribute: {node.attr}")
                return
        if isinstance(node, ast.Call):
            if not isinstance(node.func, (ast.Name, ast.Attribute)):
                self.ok, self.err = (False, "Forbidden call form (non-Name/Attribute callee)")
                return
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name) and node.value.id in SAFE_BUILTINS:
                self.ok, self.err = (False, "Forbidden subscript on builtin")
                return
        super().generic_visit(node)

def validate_expr(expr: str, extra: Optional[Set[str]] = None) -> Tuple[bool, str]:
    """PHASE A: validate expression with safe names only."""
    try:
        extra = extra or set()
        allowed = set(SAFE_FUNCS.keys()) | set(SAFE_BUILTINS.keys()) | set(SAFE_VARS) | set(extra)
        tree = ast.parse(expr, mode="eval")
        v = ExprValidator(allowed)
        v.visit(tree)
        return (v.ok, v.err or "")
    except Exception as e:
        return (False, str(e))

def safe_eval(expr: str, x: Any, extra_funcs: Optional[Dict[str, Callable]] = None) -> Any:
    """PHASE A: safe evaluation of expressions with optional helper functions."""
    ok, _ = validate_expr(expr, extra=set(extra_funcs or {}))
    if not ok:
        return float("nan")
    try:
        env: Dict[str, Any] = {}
        env.update(SAFE_FUNCS)
        env.update(SAFE_BUILTINS)
        if extra_funcs:
            env.update(extra_funcs)
        env["x"] = x
        for i in range(10):
            env[f"v{i}"] = x
        return eval(compile(ast.parse(expr, mode="eval"), "<expr>", "eval"), {"__builtins__": {}}, env)
    except Exception:
        return float("nan")


def node_count(code: str) -> int:
    try:
        return sum(1 for _ in ast.walk(ast.parse(code)))
    except Exception:
        return 999

def ast_depth(code: str) -> int:
    try:
        tree = ast.parse(code)
    except Exception:
        return 0
    max_depth = 0
    stack = [(tree, 1)]
    while stack:
        node, depth = stack.pop()
        max_depth = max(max_depth, depth)
        for child in ast.iter_child_nodes(node):
            stack.append((child, depth + 1))
    return max_depth


def program_limits_ok(code: str, max_nodes: int = 200, max_depth: int = 20, max_locals: int = 16) -> bool:
    try:
        tree = ast.parse(code)
    except Exception:
        return False
    nodes = sum(1 for _ in ast.walk(tree))
    depth = ast_depth(code)
    locals_set = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    return nodes <= max_nodes and depth <= max_depth and len(locals_set) <= max_locals


def safe_exec(code: str, x: Any, timeout_steps: int = 1000, extra_env: Optional[Dict[str, Any]] = None) -> Any:
    """Execute candidate code with step limit. Code must define run(x). Returns Any (float/list/grid)."""
    try:
        tree = ast.parse(code)
        transformer = StepLimitTransformer(timeout_steps)
        tree = transformer.visit(tree)
        ast.fix_missing_locations(tree)

        env: Dict[str, Any] = {"_steps": 0, "StepLimitExceeded": StepLimitExceeded}
        env.update(SAFE_FUNCS)
        env.update(SAFE_BUILTINS)
        if extra_env:
            env.update(extra_env)

        exec(compile(tree, "<lgp>", "exec"), {"__builtins__": {}}, env)
        if "run" not in env:
            return float("nan")
        return env["run"](x)
    except StepLimitExceeded:
        return float("nan")
    except Exception:
        return float("nan")


def safe_exec_algo(
    code: str,
    inp: Any,
    timeout_steps: int = 2000,
    max_runtime_ms: int = 50,
    extra_env: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, int, bool]:
    """Execute algo candidate code with strict step/time limits."""
    start = time.time()
    try:
        tree = ast.parse(code)
        transformer = StepLimitTransformer(timeout_steps)
        tree = transformer.visit(tree)
        ast.fix_missing_locations(tree)

        env: Dict[str, Any] = {"_steps": 0, "StepLimitExceeded": StepLimitExceeded}
        env.update(SAFE_ALGO_FUNCS)
        if extra_env:
            env.update(extra_env)

        exec(compile(tree, "<algo>", "exec"), {"__builtins__": {}}, env)
        if "run" not in env:
            return (None, env.get("_steps", 0), True)
        out = env["run"](inp)
        elapsed_ms = int((time.time() - start) * 1000)
        timed_out = elapsed_ms > max_runtime_ms
        return (out, int(env.get("_steps", 0)), timed_out)
    except StepLimitExceeded:
        return (None, int(env.get("_steps", 0) if "env" in locals() else 0), True)
    except Exception:
        return (None, int(env.get("_steps", 0) if "env" in locals() else 0), True)


def safe_exec_engine(code: str, context: Dict[str, Any], timeout_steps: int = 5000) -> Any:
    """Execute meta-engine code (selection/crossover) with safety limits."""
    try:
        tree = ast.parse(str(code))
        transformer = StepLimitTransformer(timeout_steps)
        tree = transformer.visit(tree)
        ast.fix_missing_locations(tree)

        env: Dict[str, Any] = {"_steps": 0, "StepLimitExceeded": StepLimitExceeded}
        env.update({
            "random": random,
            "math": math,
            "max": max,
            "min": min,
            "len": len,
            "sum": sum,
            "sorted": sorted,
            "int": int,
            "float": float,
            "list": list,
        })
        env.update(context)

        exec(compile(tree, "<engine>", "exec"), env)
        if "run" in env:
            return env["run"]()
        return None
    except Exception:
        return None


def safe_load_module(code: str, timeout_steps: int = 5000) -> Optional[Dict[str, Any]]:
    """PHASE B: safely load a learner module with a restricted environment."""
    ok, err = validate_code(code)
    if not ok:
        return None
    try:
        tree = ast.parse(code)
        transformer = StepLimitTransformer(timeout_steps)
        tree = transformer.visit(tree)
        ast.fix_missing_locations(tree)
        env: Dict[str, Any] = {"_steps": 0, "StepLimitExceeded": StepLimitExceeded}
        env.update(SAFE_FUNCS)
        env.update(SAFE_BUILTINS)
        exec(compile(tree, "<learner>", "exec"), {"__builtins__": {}}, env)
        return env
    except Exception:
        return None


# ---------------------------
# Engine strategy (meta-evolvable selection/crossover policy)
# ---------------------------

@dataclass
class EngineStrategy:
    selection_code: str
    crossover_code: str
    mutation_policy_code: str
    gid: str = "default"


DEFAULT_SELECTION_CODE = """
def run():
    # Context injected: pool, scores, pop_size, rng, map_elites
    # Returns: (elites, breeding_parents)
    scored = sorted(zip(pool, scores), key=lambda x: x[1])
    elite_k = max(4, pop_size // 10)
    elites = [g for g, s in scored[:elite_k]]

    parents = []
    n_needed = pop_size - len(elites)
    for _ in range(n_needed):
        # 10% chance to pick from MAP-Elites
        if rng.random() < 0.1 and map_elites and map_elites.grid:
            p = map_elites.sample(rng) or rng.choice(elites)
        else:
            p = rng.choice(elites)
        parents.append(p)
    return elites, parents
"""

DEFAULT_CROSSOVER_CODE = """
def run():
    # Context: p1 (stmts), p2 (stmts), rng
    if len(p1) < 2 or len(p2) < 2:
        return p1
    idx_a = rng.randint(0, len(p1))
    idx_b = rng.randint(0, len(p2))
    return p1[:idx_a] + p2[idx_b:]
"""

DEFAULT_MUTATION_CODE = """
def run():
    return "default"
"""


# ---------------------------
# Tasks / Datasets
# ---------------------------

@dataclass
class TaskDescriptor:
    name: str
    family: str
    input_kind: str
    output_kind: str
    n_train: int
    n_hold: int
    n_test: int
    noise: float
    stress_mult: float
    has_switch: bool
    nonlinear: bool

    def vector(self) -> List[float]:
        family_map = {
            "poly": 0.1,
            "piecewise": 0.3,
            "rational": 0.5,
            "switching": 0.7,
            "classification": 0.9,
            "list": 0.2,
            "arc": 0.4,
            "other": 0.6,
        }
        return [
            family_map.get(self.family, 0.0),
            1.0 if self.input_kind == "list" else 0.0,
            1.0 if self.input_kind == "grid" else 0.0,
            1.0 if self.output_kind == "class" else 0.0,
            float(self.n_train) / 100.0,
            float(self.n_hold) / 100.0,
            float(self.n_test) / 100.0,
            clamp(self.noise, 0.0, 1.0),
            clamp(self.stress_mult / 5.0, 0.0, 2.0),
            1.0 if self.has_switch else 0.0,
            1.0 if self.nonlinear else 0.0,
        ]

    def snapshot(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TaskSpec:
    name: str = "poly2"
    x_min: float = -3.0
    x_max: float = 3.0
    n_train: int = 96
    n_hold: int = 96
    n_test: int = 96
    noise: float = 0.01
    stress_mult: float = 3.0
    target_code: Optional[str] = None
    descriptor: Optional[TaskDescriptor] = None

    def ensure_descriptor(self) -> TaskDescriptor:
        if self.descriptor:
            return self.descriptor
        family = "other"
        if self.name in ("poly2", "poly3"):
            family = "poly"
        elif self.name == "piecewise":
            family = "piecewise"
        elif self.name == "rational":
            family = "rational"
        elif self.name == "switching":
            family = "switching"
        elif self.name == "classification":
            family = "classification"
        elif self.name in ("sort", "reverse", "filter", "max", "even_reverse_sort"):
            family = "list"
        elif self.name == "self_audit":
            family = "self_audit"
        elif self.name in ALGO_TASK_NAMES:
            family = "algo"
        elif self.name.startswith("arc_"):
            family = "arc"
        self.descriptor = TaskDescriptor(
            name=self.name,
            family=family,
            input_kind="vector"
            if family == "self_audit"
            else ("list" if family in ("list", "algo") else ("grid" if family == "arc" else "scalar")),
            output_kind="class" if family == "classification" else "scalar",
            n_train=self.n_train,
            n_hold=self.n_hold,
            n_test=self.n_test,
            noise=self.noise,
            stress_mult=self.stress_mult,
            has_switch=self.name == "switching",
            nonlinear=family in ("poly", "piecewise", "rational", "switching"),
        )
        return self.descriptor


# ---------------------------
# Algorithmic task suite (algo mode)
# ---------------------------

ALGO_TASK_NAMES = {
    "sort_int_list",
    "topk",
    "two_sum",
    "balanced_parens",
    "gcd_list",
    "rpn_eval",
    "bfs_shortest_path",
    "coin_change_min",
    "substring_find",
    "unique_count",
    "lis_length",
    "min_path_sum",
    "edit_distance",
}

ALGO_COUNTEREXAMPLES: Dict[str, List[Tuple[Any, Any]]] = {name: [] for name in ALGO_TASK_NAMES}

def _gen_int_list(rng: random.Random, min_len: int, max_len: int, lo: int = -9, hi: int = 9) -> List[int]:
    ln = rng.randint(min_len, max_len)
    return [rng.randint(lo, hi) for _ in range(ln)]

def _gen_parens(rng: random.Random, min_len: int, max_len: int) -> List[int]:
    ln = rng.randint(min_len, max_len)
    return [0 if rng.random() < 0.5 else 1 for _ in range(ln)]

def _gen_graph(rng: random.Random, n_min: int, n_max: int) -> List[List[int]]:
    n = rng.randint(n_min, n_max)
    g = []
    for i in range(n):
        neigh = []
        for j in range(n):
            if i != j and rng.random() < 0.25:
                neigh.append(j)
        g.append(neigh)
    return g

def _algo_descriptor(name: str) -> Dict[str, Any]:
    return {
        "name": name,
        "family": "algo",
        "input_kind": "list",
        "output_kind": "scalar",
        "n_train": 0,
        "n_hold": 0,
        "n_test": 0,
        "noise": 0.0,
        "stress_mult": 2.0,
        "has_switch": False,
        "nonlinear": True,
    }

def _algo_task_data(name: str, rng: random.Random, n: int, stress: bool = False) -> Tuple[List[Any], List[Any]]:
    xs: List[Any] = []
    ys: List[Any] = []
    for _ in range(n):
        if name == "sort_int_list":
            x = _gen_int_list(rng, 2, 8 if not stress else 12)
            y = sorted(x)
        elif name == "topk":
            arr = _gen_int_list(rng, 2, 10 if not stress else 14)
            k = rng.randint(1, max(1, len(arr) // 2))
            x = [arr, k]
            y = sorted(arr, reverse=True)[:k]
        elif name == "two_sum":
            arr = _gen_int_list(rng, 2, 10 if not stress else 14)
            i, j = rng.sample(range(len(arr)), 2)
            target = arr[i] + arr[j]
            x = [arr, target]
            y = [i, j]
        elif name == "balanced_parens":
            seq = _gen_parens(rng, 2, 12 if not stress else 18)
            bal = 0
            ok = 1
            for t in seq:
                bal += 1 if t == 0 else -1
                if bal < 0:
                    ok = 0
                    break
            if bal != 0:
                ok = 0
            x = seq
            y = ok
        elif name == "gcd_list":
            arr = [abs(v) + 1 for v in _gen_int_list(rng, 2, 8 if not stress else 12, 1, 9)]
            g = arr[0]
            for v in arr[1:]:
                g = math.gcd(g, v)
            x = arr
            y = g
        elif name == "rpn_eval":
            a, b = rng.randint(1, 9), rng.randint(1, 9)
            op = rng.choice([-1, -2, -3, -4])
            if op == -1:
                y = a + b
            elif op == -2:
                y = a - b
            elif op == -3:
                y = a * b
            else:
                y = a // b if b else 0
            x = [a, b, op]
        elif name == "bfs_shortest_path":
            g = _gen_graph(rng, 4, 7 if not stress else 9)
            s, t = rng.sample(range(len(g)), 2)
            dist = [-1] * len(g)
            dist[s] = 0
            q = [s]
            while q:
                cur = q.pop(0)
                for nxt in g[cur]:
                    if dist[nxt] == -1:
                        dist[nxt] = dist[cur] + 1
                        q.append(nxt)
            x = [g, s, t]
            y = dist[t]
        elif name == "coin_change_min":
            coins = [c for c in _gen_int_list(rng, 2, 5 if not stress else 7, 1, 8) if c > 0]
            amount = rng.randint(1, 12 if not stress else 18)
            dp = [float("inf")] * (amount + 1)
            dp[0] = 0
            for c in coins:
                for a in range(c, amount + 1):
                    dp[a] = min(dp[a], dp[a - c] + 1)
            y = -1 if dp[amount] == float("inf") else int(dp[amount])
            x = [coins, amount]
        elif name == "substring_find":
            hay = _gen_int_list(rng, 4, 10 if not stress else 14, 1, 4)
            needle = hay[1:3] if len(hay) > 3 and rng.random() < 0.7 else _gen_int_list(rng, 2, 3, 1, 4)
            idx = -1
            for i in range(len(hay) - len(needle) + 1):
                if hay[i:i + len(needle)] == needle:
                    idx = i
                    break
            x = [hay, needle]
            y = idx
        elif name == "unique_count":
            arr = _gen_int_list(rng, 3, 10 if not stress else 14, 1, 6)
            x = arr
            y = len(set(arr))
        elif name == "lis_length":
            arr = _gen_int_list(rng, 3, 10 if not stress else 14, -5, 9)
            dp = [1 for _ in arr]
            for i in range(len(arr)):
                for j in range(i):
                    if arr[j] < arr[i]:
                        dp[i] = max(dp[i], dp[j] + 1)
            x = arr
            y = max(dp) if dp else 0
        elif name == "min_path_sum":
            rows = rng.randint(2, 5 if not stress else 7)
            cols = rng.randint(2, 5 if not stress else 7)
            grid = [[rng.randint(0, 9) for _ in range(cols)] for _ in range(rows)]
            dp = [[0 for _ in range(cols)] for _ in range(rows)]
            dp[0][0] = grid[0][0]
            for r in range(1, rows):
                dp[r][0] = dp[r - 1][0] + grid[r][0]
            for c in range(1, cols):
                dp[0][c] = dp[0][c - 1] + grid[0][c]
            for r in range(1, rows):
                for c in range(1, cols):
                    dp[r][c] = min(dp[r - 1][c], dp[r][c - 1]) + grid[r][c]
            x = grid
            y = dp[-1][-1]
        elif name == "edit_distance":
            a = _gen_int_list(rng, 2, 6 if not stress else 8, 0, 4)
            b = _gen_int_list(rng, 2, 6 if not stress else 8, 0, 4)
            dp = [[0 for _ in range(len(b) + 1)] for _ in range(len(a) + 1)]
            for i in range(len(a) + 1):
                dp[i][0] = i
            for j in range(len(b) + 1):
                dp[0][j] = j
            for i in range(1, len(a) + 1):
                for j in range(1, len(b) + 1):
                    cost = 0 if a[i - 1] == b[j - 1] else 1
                    dp[i][j] = min(
                        dp[i - 1][j] + 1,
                        dp[i][j - 1] + 1,
                        dp[i - 1][j - 1] + cost,
                    )
            x = [a, b]
            y = dp[-1][-1]
        else:
            x = []
            y = 0
        xs.append(x)
        ys.append(y)
    return xs, ys

def algo_batch(name: str, seed: int, freeze_eval: bool = True, train_resample_every: int = 1, gen: int = 0) -> Optional[Batch]:
    if name not in ALGO_TASK_NAMES:
        return None
    rng = random.Random(seed)
    hold_rng = random.Random(seed + 11)
    stress_rng = random.Random(seed + 29)
    test_rng = random.Random(seed + 47)
    if not freeze_eval:
        hold_rng = random.Random(seed + 11 + gen)
        stress_rng = random.Random(seed + 29 + gen)
        test_rng = random.Random(seed + 47 + gen)
    train_rng = rng if train_resample_every <= 1 else random.Random(seed + gen // max(1, train_resample_every))
    x_tr, y_tr = _algo_task_data(name, train_rng, 40, stress=False)
    x_ho, y_ho = _algo_task_data(name, hold_rng, 24, stress=False)
    x_st, y_st = _algo_task_data(name, stress_rng, 24, stress=True)
    x_te, y_te = _algo_task_data(name, test_rng, 24, stress=True)
    return Batch(x_tr, y_tr, x_ho, y_ho, x_st, y_st, x_te, y_te)


@dataclass
class ControlPacket:
    mutation_rate: Optional[float] = None
    crossover_rate: Optional[float] = None
    novelty_weight: float = 0.0
    branch_insert_rate: float = 0.0
    op_weights: Optional[Dict[str, float]] = None
    acceptance_margin: float = 1e-9
    patience: int = 5

    def get(self, key: str, default: Any = None) -> Any:
        val = getattr(self, key, default)
        if val is None:
            return default
        return val


TARGET_FNS = {
    "sort": lambda x: sorted(x),
    "reverse": lambda x: list(reversed(x)),
    "max": lambda x: max(x) if x else 0,
    "filter": lambda x: [v for v in x if v > 0],
    "arc_ident": lambda x: x,
    "arc_rot90": lambda x: [list(r) for r in zip(*x[::-1])],
    "arc_inv": lambda x: [[1 - c if c in (0, 1) else c for c in r] for r in x],
    "poly2": lambda x: 0.7 * x * x - 0.2 * x + 0.3,
    "poly3": lambda x: 0.3 * x ** 3 - 0.5 * x + 0.1,
    "piecewise": lambda x: (-0.5 * x + 1.0) if x < 0 else (0.3 * x * x + 0.1),
    "rational": lambda x: (x * x + 1.0) / (1.0 + 0.5 * abs(x)),
    "sinmix": lambda x: math.sin(x) + 0.3 * math.cos(2 * x),
    "absline": lambda x: abs(x) + 0.2 * x,
    "classification": lambda x: 1.0 if (x + 0.25 * math.sin(3 * x)) > 0 else 0.0,
}


ARC_GYM_PATH = os.path.join(os.path.dirname(__file__), "ARC_GYM")

def load_arc_task(task_id: str) -> Dict:
    fname = task_id
    if not fname.endswith(".json"):
        fname += ".json"
    path = os.path.join(ARC_GYM_PATH, fname)
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_arc_tasks() -> List[str]:
    if not os.path.exists(ARC_GYM_PATH):
        return []
    return [f[:-5] for f in os.listdir(ARC_GYM_PATH) if f.endswith(".json")]

@dataclass
class Batch:
    x_tr: List[Any]
    y_tr: List[Any]
    x_ho: List[Any]
    y_ho: List[Any]
    x_st: List[Any]
    y_st: List[Any]
    x_te: List[Any]
    y_te: List[Any]


def _best_code_snapshot() -> str:
    try:
        state = load_state()
    except Exception:
        state = None
    if not state or not state.universes:
        return "def run(x):\n    return x\n"
    target = next((u for u in state.universes if u.get("uid") == state.selected_uid), None)
    if not target:
        target = state.universes[0]
    best = target.get("best")
    if not best:
        return "def run(x):\n    return x\n"
    if state.mode == "learner":
        return LearnerGenome(**best).code
    return Genome(**best).code


def _code_features(code: str) -> List[float]:
    return [
        float(len(code)),
        float(node_count(code)),
        float(code.count("if ")),
        float(code.count("while ")),
        float(code.count("return ")),
    ]

def sample_batch(rng: random.Random, t: TaskSpec) -> Optional[Batch]:
    # function target
    if t.target_code:
        f = lambda x: safe_exec(t.target_code, x)
    elif t.name in ("sort", "reverse", "filter", "max"):
        f = TARGET_FNS.get(t.name) or (lambda x: sorted(x))
    else:
        f = TARGET_FNS.get(t.name, lambda x: x)

    if t.name == "self_audit":
        base_code = _best_code_snapshot()
        base_features = _code_features(base_code)

        def synth_features(k: int, jitter: float) -> List[List[float]]:
            samples = []
            for _ in range(k):
                sample = [max(0.0, f + rng.gauss(0, jitter)) for f in base_features]
                samples.append(sample)
            return samples

        def target_score(vec: List[float]) -> float:
            length, nodes, ifs, whiles, returns = vec
            raw = 0.004 * length + 0.02 * nodes + 0.1 * ifs + 0.15 * whiles + 0.02 * returns
            return math.tanh(raw / 10.0)

        x_tr = synth_features(t.n_train, 3.0)
        x_ho = synth_features(t.n_hold, 4.0)
        x_st = synth_features(t.n_hold, 6.0)
        x_te = synth_features(t.n_test, 4.0)
        y_tr = [target_score(x) for x in x_tr]
        y_ho = [target_score(x) for x in x_ho]
        y_st = [target_score(x) for x in x_st]
        y_te = [target_score(x) for x in x_te]
        return Batch(x_tr, y_tr, x_ho, y_ho, x_st, y_st, x_te, y_te)

    # ARC tasks from local json
    json_data = load_arc_task(t.name.replace("arc_", ""))
    if json_data:
        pairs = json_data.get("train", []) + json_data.get("test", [])
        x_all, y_all = [], []
        for p in pairs:
            x_all.append(p["input"])
            y_all.append(p["output"])
            if len(x_all) >= 30:
                break
        if not x_all:
            return None
        return Batch(
            x_all[:20], y_all[:20],
            x_all[:10], y_all[:10],
            x_all[:5],  y_all[:5],
            x_all[5:10], y_all[5:10],
        )

    # list tasks
    def gen_lists(k, min_len, max_len):
        data = []
        for _ in range(k):
            a = max(1, int(min_len))
            b = max(a, int(max_len))
            l = rng.randint(a, b)
            data.append([rng.randint(-100, 100) for _ in range(l)])
        return data

    if t.name == "even_reverse_sort":
        f = lambda x: sorted([n for n in x if n % 2 == 0], reverse=True)
        x_tr = gen_lists(t.n_train, t.x_min, t.x_max)
        x_ho = gen_lists(t.n_hold, t.x_min + 2, t.x_max + 2)
        x_st = gen_lists(t.n_hold, t.x_max + 5, t.x_max + 10)
        y_tr = [f(x) for x in x_tr]
        y_ho = [f(x) for x in x_ho]
        y_st = [f(x) for x in x_st]
        x_te = gen_lists(max(1, t.n_test), t.x_min + 1, t.x_max + 1)
        y_te = [f(x) for x in x_te]
        return Batch(x_tr, y_tr, x_ho, y_ho, x_st, y_st, x_te, y_te)

    if t.name in ("sort", "reverse", "filter", "max"):
        x_tr = gen_lists(t.n_train, t.x_min, t.x_max)
        x_ho = gen_lists(t.n_hold, t.x_min + 2, t.x_max + 2)
        x_st = gen_lists(t.n_hold, t.x_max + 5, t.x_max + 10)
        y_tr = [f(x) for x in x_tr]
        y_ho = [f(x) for x in x_ho]
        y_st = [f(x) for x in x_st]
        x_te = gen_lists(max(1, t.n_test), t.x_min + 1, t.x_max + 1)
        y_te = [f(x) for x in x_te]
        return Batch(x_tr, y_tr, x_ho, y_ho, x_st, y_st, x_te, y_te)

    # synthetic ARC-like generators if name starts with arc_
    if t.name.startswith("arc_"):
        def gen_grids(k, dim):
            data = []
            for _ in range(k):
                g = [[rng.randint(0, 1) for _ in range(dim)] for _ in range(dim)]
                data.append(g)
            return data
        dim = int(t.x_min) if t.x_min > 0 else 3
        x_tr = gen_grids(20, dim)
        x_ho = gen_grids(10, dim)
        x_st = gen_grids(10, dim + 1)
        y_tr = [f(x) for x in x_tr]
        y_ho = [f(x) for x in x_ho]
        y_st = [f(x) for x in x_st]
        x_te = gen_grids(10, dim)
        y_te = [f(x) for x in x_te]
        return Batch(x_tr, y_tr, x_ho, y_ho, x_st, y_st, x_te, y_te)

    if t.name == "switching":
        def target_switch(pair):
            x, s = pair
            return TARGET_FNS["poly2"](x) if s < 0.5 else TARGET_FNS["sinmix"](x)

        def gen_pairs(k, a, b):
            data = []
            for _ in range(k):
                x = a + (b - a) * rng.random()
                s = 1.0 if rng.random() > 0.5 else 0.0
                data.append([x, s])
            return data

        x_tr = gen_pairs(t.n_train, t.x_min, t.x_max)
        x_ho = gen_pairs(t.n_hold, t.x_min, t.x_max)
        x_st = gen_pairs(t.n_hold, t.x_min * t.stress_mult, t.x_max * t.stress_mult)
        x_te = gen_pairs(t.n_test, t.x_min, t.x_max)
        y_tr = [target_switch(x) for x in x_tr]
        y_ho = [target_switch(x) for x in x_ho]
        y_st = [target_switch(x) for x in x_st]
        y_te = [target_switch(x) for x in x_te]
        return Batch(x_tr, y_tr, x_ho, y_ho, x_st, y_st, x_te, y_te)

    # numeric regression tasks
    xs = lambda n, a, b: [a + (b - a) * rng.random() for _ in range(n)]
    ys = lambda xv, n: [f(x) + rng.gauss(0, n) if n > 0 else f(x) for x in xv]
    half = 0.5 * (t.x_max - t.x_min)
    mid = 0.5 * (t.x_min + t.x_max)
    x_tr = xs(t.n_train, t.x_min, t.x_max)
    x_ho = xs(t.n_hold, t.x_min, t.x_max)
    x_st = xs(t.n_hold, mid - half * t.stress_mult, mid + half * t.stress_mult)
    x_te = xs(t.n_test, t.x_min, t.x_max)
    return Batch(
        x_tr, ys(x_tr, t.noise),
        x_ho, ys(x_ho, t.noise),
        x_st, ys(x_st, t.noise * t.stress_mult),
        x_te, ys(x_te, t.noise),
    )


def task_suite(seed: int) -> List[TaskSpec]:
    base = [
        TaskSpec(name="poly2", x_min=-3.0, x_max=3.0, n_train=96, n_hold=64, n_test=64, noise=0.01),
        TaskSpec(name="poly3", x_min=-4.0, x_max=4.0, n_train=96, n_hold=64, n_test=64, noise=0.01),
        TaskSpec(name="piecewise", x_min=-4.0, x_max=4.0, n_train=96, n_hold=64, n_test=64, noise=0.01),
        TaskSpec(name="rational", x_min=-5.0, x_max=5.0, n_train=96, n_hold=64, n_test=64, noise=0.02),
        TaskSpec(name="switching", x_min=-3.0, x_max=3.0, n_train=96, n_hold=64, n_test=64, noise=0.0),
        TaskSpec(name="classification", x_min=-4.0, x_max=4.0, n_train=96, n_hold=64, n_test=64, noise=0.0),
        TaskSpec(name="sinmix", x_min=-6.0, x_max=6.0, n_train=96, n_hold=64, n_test=64, noise=0.01),
        TaskSpec(name="absline", x_min=-6.0, x_max=6.0, n_train=96, n_hold=64, n_test=64, noise=0.01),
        TaskSpec(name="self_audit", x_min=0.0, x_max=1.0, n_train=64, n_hold=48, n_test=48, noise=0.0),
    ]
    rng = random.Random(seed)
    rng.shuffle(base)
    return base


def split_meta_tasks(seed: int, meta_train_ratio: float = 0.6) -> Tuple[List[TaskSpec], List[TaskSpec]]:
    suite = task_suite(seed)
    cut = max(1, int(len(suite) * meta_train_ratio))
    return suite[:cut], suite[cut:]


FROZEN_BATCH_CACHE: Dict[str, Batch] = {}


def _task_cache_key(task: TaskSpec, seed: int) -> str:
    return f"{task.name}:{seed}:{task.x_min}:{task.x_max}:{task.n_train}:{task.n_hold}:{task.n_test}:{task.noise}:{task.stress_mult}:{task.target_code}"


def get_task_batch(
    task: TaskSpec,
    seed: int,
    freeze_eval: bool = True,
    train_resample_every: int = 1,
    gen: int = 0,
) -> Optional[Batch]:
    if task.name in ALGO_TASK_NAMES:
        return algo_batch(task.name, seed, freeze_eval=freeze_eval, train_resample_every=train_resample_every, gen=gen)
    key = _task_cache_key(task, seed)
    if freeze_eval and key in FROZEN_BATCH_CACHE:
        return FROZEN_BATCH_CACHE[key]
    h = int(sha256(key)[:8], 16)
    rng = random.Random(h if freeze_eval else seed)
    batch = sample_batch(rng, task)
    if freeze_eval and batch is not None:
        FROZEN_BATCH_CACHE[key] = batch
    return batch


# ---------------------------
# Genome / Evaluation
# ---------------------------

@dataclass
class Genome:
    statements: List[str]
    gid: str = ""
    parents: List[str] = field(default_factory=list)
    op_tag: str = "init"
    birth_ms: int = 0

    @property
    def code(self) -> str:
        body = "\n    ".join(self.statements) if self.statements else "return x"
        return f"def run(x):\n    # {self.gid}\n    v0=x\n    {body}"

    def __post_init__(self):
        if not self.gid:
            self.gid = sha256("".join(self.statements) + str(time.time()))[:12]
        if not self.birth_ms:
            self.birth_ms = now_ms()


@dataclass
class LearnerGenome:
    """PHASE B: learner genome with encode/predict/update/objective blocks."""
    encode_stmts: List[str]
    predict_stmts: List[str]
    update_stmts: List[str]
    objective_stmts: List[str]
    gid: str = ""
    parents: List[str] = field(default_factory=list)
    op_tag: str = "init"
    birth_ms: int = 0

    @property
    def code(self) -> str:
        def ensure_return(stmts: List[str], fallback: str) -> List[str]:
            for s in stmts:
                if s.strip().startswith("return "):
                    return stmts
            return stmts + [fallback]

        enc = ensure_return(self.encode_stmts or [], "return x")
        pred = ensure_return(self.predict_stmts or [], "return z")
        upd = ensure_return(self.update_stmts or [], "return mem")
        obj = ensure_return(self.objective_stmts or [], "return hold + 0.5*stress + 0.01*nodes")

        enc_body = "\n    ".join(enc) if enc else "return x"
        pred_body = "\n    ".join(pred) if pred else "return z"
        upd_body = "\n    ".join(upd) if upd else "return mem"
        obj_body = "\n    ".join(obj) if obj else "return hold + 0.5*stress + 0.01*nodes"

        return (
            "def init_mem():\n"
            "    return {\"w\": 0.0, \"b\": 0.0, \"t\": 0}\n\n"
            "def encode(x, mem):\n"
            f"    # {self.gid}\n    {enc_body}\n\n"
            "def predict(z, mem):\n"
            f"    {pred_body}\n\n"
            "def update(mem, x, y_pred, y_true, lr=0.05):\n"
            f"    {upd_body}\n\n"
            "def objective(train, hold, stress, nodes):\n"
            f"    {obj_body}\n"
        )

    def __post_init__(self):
        if not self.gid:
            self.gid = sha256("".join(self.encode_stmts + self.predict_stmts + self.update_stmts + self.objective_stmts) + str(time.time()))[:12]
        if not self.birth_ms:
            self.birth_ms = now_ms()


@dataclass
class EvalResult:
    ok: bool
    train: float
    hold: float
    stress: float
    test: float
    nodes: int
    score: float
    err: Optional[str] = None


SCORE_W_HOLD = 0.452390
SCORE_W_STRESS = 0.4
SCORE_W_TRAIN = 0.0


def calc_error(p: Any, t: Any) -> float:
    if isinstance(t, (int, float)):
        if isinstance(p, (int, float)):
            return (p - t) ** 2
        return 1_000_000.0
    if isinstance(t, list):
        if not isinstance(p, list):
            return 1_000_000.0
        if len(p) != len(t):
            return 1000.0 * abs(len(p) - len(t))
        return sum(calc_error(pv, tv) for pv, tv in zip(p, t))
    return 1_000_000.0


def _list_invariance_penalty(x: Any, p: Any, task_name: str) -> float:
    if not isinstance(x, list):
        return 0.0
    if task_name in ("sort", "reverse"):
        if not isinstance(p, list):
            return 5_000.0
        if len(p) != len(x):
            return 2_000.0 + 10.0 * abs(len(p) - len(x))
        try:
            if collections.Counter(p) != collections.Counter(x):
                return 2_000.0
        except TypeError:
            pass
    if task_name == "filter":
        if not isinstance(p, list):
            return 5_000.0
        try:
            x_counts = collections.Counter(x)
            p_counts = collections.Counter(p)
            for k, v in p_counts.items():
                if x_counts.get(k, 0) < v:
                    return 2_000.0
        except TypeError:
            pass
    if task_name == "max":
        if not isinstance(p, (int, float)):
            return 5_000.0
    return 0.0


def calc_loss_sort(p: List[Any], t: List[Any]) -> float:
    if not isinstance(p, list):
        return 1_000_000.0
    if len(p) != len(t):
        return 1000.0 * abs(len(p) - len(t))
    p_sorted = sorted(p) if all(isinstance(x, (int, float)) for x in p) else p
    t_sorted = sorted(t)
    content_loss = sum((a - b) ** 2 for a, b in zip(p_sorted, t_sorted))
    if content_loss > 0.1:
        return 1000.0 + content_loss
    inversions = 0
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            if p[i] > p[j]:
                inversions += 1
    return float(inversions)


def calc_heuristic_loss(p: Any, t: Any, task_name: str, x: Any = None) -> float:
    penalty = _list_invariance_penalty(x, p, task_name)
    if task_name == "sort":
        return calc_loss_sort(p, t) + penalty
    if isinstance(t, list):
        if not isinstance(p, list):
            return 1_000_000.0 + penalty
        if len(p) != len(t):
            return 500.0 * abs(len(p) - len(t)) + penalty
        if task_name in ("reverse", "filter"):
            return sum(calc_error(pv, tv) for pv, tv in zip(p, t)) + penalty
    if task_name.startswith("arc_"):
        if not isinstance(p, list) or not p or not isinstance(p[0], list):
            return 1000.0 + penalty
        if len(p) != len(t) or len(p[0]) != len(t[0]):
            return 500.0 + abs(len(p) - len(t)) + abs(len(p[0]) - len(t[0])) + penalty
        err = 0
        for r in range(len(t)):
            for c in range(len(t[0])):
                if p[r][c] != t[r][c]:
                    err += 1
        return float(err) + penalty
    return calc_error(p, t) + penalty


def mse_exec(
    code: str,
    xs: List[Any],
    ys: List[Any],
    task_name: str = "",
    extra_env: Optional[Dict[str, Any]] = None,
    validator: Callable[[str], Tuple[bool, str]] = validate_code,
) -> Tuple[bool, float, str]:
    ok, err = validator(code)
    if not ok:
        return (False, float("inf"), err)
    if validator == validate_program and not program_limits_ok(code):
        return (False, float("inf"), "program_limits")
    try:
        total_err = 0.0
        for x, y in zip(xs, ys):
            pred = safe_exec(code, x, extra_env=extra_env)
            if pred is None:
                return (False, float("inf"), "No return")
            if task_name in ("sort", "reverse", "max", "filter") or task_name.startswith("arc_"):
                total_err += calc_heuristic_loss(pred, y, task_name, x=x)
            else:
                total_err += calc_error(pred, y)
        return (True, total_err / max(1, len(xs)), "")
    except Exception as e:
        return (False, float("inf"), f"{type(e).__name__}: {str(e)}")


def _algo_equal(a: Any, b: Any) -> bool:
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(_algo_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return False
        return all(_algo_equal(a[k], b[k]) for k in a.keys())
    return a == b


def algo_exec(
    code: str,
    xs: List[Any],
    ys: List[Any],
    task_name: str,
    counterexamples: Optional[List[Tuple[Any, Any]]] = None,
    validator: Callable[[str], Tuple[bool, str]] = validate_algo_program,
) -> Tuple[bool, float, int, float, int, str]:
    ok, err = validator(code)
    if not ok:
        return (False, 1.0, 0, 1.0, 0, err)
    total = 0
    timeouts = 0
    steps = 0
    failures = 0
    extra = counterexamples[:] if counterexamples else []
    xs_all = list(xs) + [x for x, _ in extra]
    ys_all = list(ys) + [y for _, y in extra]
    for x, y in zip(xs_all, ys_all):
        out, used, timeout = safe_exec_algo(code, x)
        steps += used
        if timeout:
            timeouts += 1
        if not _algo_equal(out, y):
            failures += 1
            if counterexamples is not None and len(counterexamples) < 64:
                counterexamples.append((x, y))
        total += 1
    err_rate = failures / max(1, total)
    timeout_rate = timeouts / max(1, total)
    avg_steps = steps // max(1, total)
    return (True, err_rate, avg_steps, timeout_rate, total, "")


def evaluate_algo(
    g: Genome,
    b: Batch,
    task_name: str,
    lam: float = 0.0001,
) -> EvalResult:
    code = g.code
    counterexamples = ALGO_COUNTEREXAMPLES.get(task_name, [])
    ok1, tr_err, tr_steps, tr_timeout, _, e1 = algo_exec(code, b.x_tr, b.y_tr, task_name, counterexamples)
    ok2, ho_err, ho_steps, ho_timeout, _, e2 = algo_exec(code, b.x_ho, b.y_ho, task_name, counterexamples)
    ok3, st_err, st_steps, st_timeout, _, e3 = algo_exec(code, b.x_st, b.y_st, task_name, counterexamples)
    ok4, te_err, te_steps, te_timeout, _, e4 = algo_exec(code, b.x_te, b.y_te, task_name, counterexamples)
    ok = ok1 and ok2 and ok3 and ok4 and all(math.isfinite(v) for v in (tr_err, ho_err, st_err, te_err))
    nodes = node_count(code)
    step_penalty = 0.0001 * (tr_steps + ho_steps + st_steps + te_steps)
    timeout_penalty = 0.5 * (tr_timeout + ho_timeout + st_timeout + te_timeout)
    if not ok:
        return EvalResult(False, tr_err, ho_err, st_err, te_err, nodes, float("inf"), e1 or e2 or e3 or e4 or "nan")
    # Hard cutoff: stress overflows are rejected before any score aggregation.
    if st_err > STRESS_MAX:
        return EvalResult(False, tr_err, ho_err, st_err, te_err, nodes, float("inf"), "stress_overflow")
    score = SCORE_W_HOLD * ho_err + SCORE_W_STRESS * st_err + SCORE_W_TRAIN * tr_err + lam * nodes + step_penalty + timeout_penalty
    err = e1 or e2 or e3 or e4
    return EvalResult(ok, tr_err, ho_err, st_err, te_err, nodes, score, err or None)


def evaluate(
    g: Genome,
    b: Batch,
    task_name: str,
    lam: float = 0.0001,
    extra_env: Optional[Dict[str, Any]] = None,
    validator: Callable[[str], Tuple[bool, str]] = validate_code,
) -> EvalResult:
    code = g.code
    ok1, tr, e1 = mse_exec(code, b.x_tr, b.y_tr, task_name, extra_env=extra_env)
    ok2, ho, e2 = mse_exec(code, b.x_ho, b.y_ho, task_name, extra_env=extra_env)
    ok3, st, e3 = mse_exec(code, b.x_st, b.y_st, task_name, extra_env=extra_env)
    ok4, te, e4 = mse_exec(code, b.x_te, b.y_te, task_name, extra_env=extra_env)
    ok = ok1 and ok2 and ok3 and ok4 and all(math.isfinite(v) for v in (tr, ho, st, te))
    nodes = node_count(code)
    if not ok:
        return EvalResult(False, tr, ho, st, te, nodes, float("inf"), e1 or e2 or e3 or e4 or "nan")
    # Hard cutoff: stress overflows are rejected before any score aggregation.
    if st > STRESS_MAX:
        return EvalResult(False, tr, ho, st, te, nodes, float("inf"), "stress_overflow")
    score = SCORE_W_HOLD * ho + SCORE_W_STRESS * st + SCORE_W_TRAIN * tr + lam * nodes
    err = e1 or e2 or e3 or e4
    return EvalResult(ok, tr, ho, st, te, nodes, score, err or None)


def evaluate_learner(
    learner: LearnerGenome,
    b: Batch,
    task_name: str,
    adapt_steps: int = 8,
    lam: float = 0.0001,
) -> EvalResult:
    """PHASE B: evaluate learner with adaptation on training only."""
    env = safe_load_module(learner.code)
    if not env:
        return EvalResult(False, float("inf"), float("inf"), float("inf"), float("inf"), 0, float("inf"), "load_failed")
    required = ["init_mem", "encode", "predict", "update", "objective"]
    if not all(name in env and callable(env[name]) for name in required):
        return EvalResult(False, float("inf"), float("inf"), float("inf"), float("inf"), 0, float("inf"), "missing_funcs")

    init_mem = env["init_mem"]
    encode = env["encode"]
    predict = env["predict"]
    update = env["update"]
    objective = env["objective"]

    try:
        mem = init_mem()
    except Exception:
        mem = {"w": 0.0, "b": 0.0, "t": 0}

    def run_eval(xs: List[Any], ys: List[Any], do_update: bool) -> float:
        nonlocal mem
        total = 0.0
        for i, (x, y) in enumerate(zip(xs, ys)):
            try:
                z = encode(x, mem)
                y_pred = predict(z, mem)
            except Exception:
                y_pred = None
            if task_name in ("sort", "reverse", "max", "filter") or task_name.startswith("arc_"):
                total += calc_heuristic_loss(y_pred, y, task_name, x=x)
            else:
                total += calc_error(y_pred, y)
            if do_update and i < adapt_steps:
                try:
                    mem = update(mem, x, y_pred, y, 0.05)
                except Exception:
                    pass
        return total / max(1, len(xs))

    try:
        train = run_eval(b.x_tr, b.y_tr, do_update=True)
        hold = run_eval(b.x_ho, b.y_ho, do_update=False)
        stress = run_eval(b.x_st, b.y_st, do_update=False)
        test = run_eval(b.x_te, b.y_te, do_update=False)
        nodes = node_count(learner.code)
        ok = all(math.isfinite(v) for v in (train, hold, stress, test))
        if not ok:
            return EvalResult(False, train, hold, stress, test, nodes, float("inf"), "nan")
        # Hard cutoff: stress overflows are rejected before any score aggregation.
        if stress > STRESS_MAX:
            return EvalResult(False, train, hold, stress, test, nodes, float("inf"), "stress_overflow")
        obj = objective(train, hold, stress, nodes)
        if not isinstance(obj, (int, float)) or not math.isfinite(obj):
            obj = SCORE_W_HOLD * hold + SCORE_W_STRESS * stress
        score = float(obj) + lam * nodes
        ok = all(math.isfinite(v) for v in (train, hold, stress, test, score))
        return EvalResult(ok, train, hold, stress, test, nodes, score, None if ok else "nan")
    except Exception as exc:
        return EvalResult(False, float("inf"), float("inf"), float("inf"), float("inf"), 0, float("inf"), str(exc))


# ---------------------------
# Mutation operators
# ---------------------------

def _pick_node(rng: random.Random, body: ast.AST) -> ast.AST:
    nodes = list(ast.walk(body))
    return rng.choice(nodes[1:]) if len(nodes) > 1 else body

def _to_src(body: ast.AST) -> str:
    try:
        return ast.unparse(body)
    except Exception:
        return "x"

def _random_expr(rng: random.Random, depth: int = 0) -> str:
    if depth > 2:
        return rng.choice(["x", "v0", str(rng.randint(0, 9))])
    options = ["binop", "call", "const", "var"]
    weights = [GRAMMAR_PROBS.get(k, 1.0) for k in options]
    mtype = rng.choices(options, weights=weights, k=1)[0]
    if mtype == "binop":
        op = rng.choice(["+", "-", "*", "/", "**", "%"])
        return f"({_random_expr(rng, depth + 1)} {op} {_random_expr(rng, depth + 1)})"
    if mtype == "call":
        funcs = list(SAFE_FUNCS.keys())
        f_weights = [GRAMMAR_PROBS.get(f, 0.5) for f in funcs]
        fname = rng.choices(funcs, weights=f_weights, k=1)[0]
        return f"{fname}({_random_expr(rng, depth + 1)})"
    if mtype == "const":
        return f"{rng.uniform(-2, 2):.2f}"
    return rng.choice(["x", "v0"])


def op_insert_assign(rng: random.Random, stmts: List[str]) -> List[str]:
    new_stmts = stmts[:]
    idx = rng.randint(0, len(new_stmts))
    var = f"v{rng.randint(0, 3)}"
    expr = _random_expr(rng)
    new_stmts.insert(idx, f"{var} = {expr}")
    return new_stmts

def op_insert_if(rng: random.Random, stmts: List[str]) -> List[str]:
    if not stmts:
        return stmts
    new_stmts = stmts[:]
    idx = rng.randint(0, len(new_stmts) - 1)
    cond = f"v{rng.randint(0, 3)} < {rng.randint(0, 10)}"
    block = [f"    {s}" for s in new_stmts[idx: idx + 2]]
    new_stmts[idx: idx + 2] = [f"if {cond}:"] + block
    return new_stmts

def op_insert_while(rng: random.Random, stmts: List[str]) -> List[str]:
    if not stmts:
        return stmts
    new_stmts = stmts[:]
    idx = rng.randint(0, len(new_stmts) - 1)
    cond = f"v{rng.randint(0, 3)} < {rng.randint(0, 10)}"
    block = [f"    {s}" for s in new_stmts[idx: idx + 2]]
    new_stmts[idx: idx + 2] = [f"while {cond}:"] + block
    return new_stmts

def op_delete_stmt(rng: random.Random, stmts: List[str]) -> List[str]:
    if not stmts:
        return stmts
    new_stmts = stmts[:]
    new_stmts.pop(rng.randint(0, len(new_stmts) - 1))
    return new_stmts

def op_modify_line(rng: random.Random, stmts: List[str]) -> List[str]:
    if not stmts:
        return stmts
    new_stmts = stmts[:]
    idx = rng.randint(0, len(new_stmts) - 1)
    if "=" in new_stmts[idx]:
        var = new_stmts[idx].split("=")[0].strip()
        new_stmts[idx] = f"{var} = {_random_expr(rng)}"
    return new_stmts

def op_tweak_const(rng: random.Random, stmts: List[str]) -> List[str]:
    if not stmts:
        return stmts
    new_stmts = stmts[:]
    idx = rng.randint(0, len(new_stmts) - 1)

    class TweakTransformer(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                val = float(node.value)
                new_val = val + rng.gauss(0, 0.1 * abs(val) + 0.01)
                if rng.random() < 0.05:
                    new_val = -val
                if rng.random() < 0.05:
                    new_val = 0.0
                return ast.Constant(value=new_val)
            return node

    try:
        tree = ast.parse(new_stmts[idx], mode="exec")
        new_tree = TweakTransformer().visit(tree)
        ast.fix_missing_locations(new_tree)
        new_stmts[idx] = ast.unparse(new_tree).strip()
    except Exception:
        pass
    return new_stmts

def op_change_binary(rng: random.Random, stmts: List[str]) -> List[str]:
    if not stmts:
        return stmts
    new_stmts = stmts[:]
    idx = rng.randint(0, len(new_stmts) - 1)
    pops = [ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod]

    class OpTransformer(ast.NodeTransformer):
        def visit_BinOp(self, node):
            node = self.generic_visit(node)
            if rng.random() < 0.5:
                node.op = rng.choice(pops)()
            return node

    try:
        tree = ast.parse(new_stmts[idx], mode="exec")
        new_tree = OpTransformer().visit(tree)
        ast.fix_missing_locations(new_tree)
        new_stmts[idx] = ast.unparse(new_tree).strip()
    except Exception:
        pass
    return new_stmts

def op_list_manipulation(rng: random.Random, stmts: List[str]) -> List[str]:
    if not stmts:
        return stmts
    new_stmts = stmts[:]
    idx = rng.randint(0, len(new_stmts))
    ops = [
        f"v{rng.randint(0,3)} = x[{rng.randint(0,2)}]",
        f"if len(x) > {rng.randint(1,5)}: v{rng.randint(0,3)} = x[0]",
        "v0, v1 = v1, v0",  # requires Tuple allowed
        f"v{rng.randint(0,3)} = sorted(x)",
    ]
    new_stmts.insert(idx, rng.choice(ops))
    return new_stmts

def op_modify_return(rng: random.Random, stmts: List[str]) -> List[str]:
    new_stmts = stmts[:]
    active_vars = ["x"] + [f"v{i}" for i in range(4)]
    for i in range(len(new_stmts) - 1, -1, -1):
        if new_stmts[i].strip().startswith("return "):
            new_stmts[i] = f"return {rng.choice(active_vars)}"
            return new_stmts
    new_stmts.append(f"return {rng.choice(active_vars)}")
    return new_stmts


def op_learner_update_step(rng: random.Random, stmts: List[str]) -> List[str]:
    new_stmts = stmts[:]
    idx = rng.randint(0, len(new_stmts))
    ops = [
        "mem['w'] = mem['w'] + lr * (y_true - y_pred) * x",
        "mem['b'] = mem['b'] + lr * (y_true - y_pred)",
        "mem['t'] = mem['t'] + 1",
        "return mem",
    ]
    new_stmts.insert(idx, rng.choice(ops))
    return new_stmts


def op_learner_objective_tweak(rng: random.Random, stmts: List[str]) -> List[str]:
    new_stmts = stmts[:]
    idx = rng.randint(0, len(new_stmts))
    expr = rng.choice([
        "return hold + 0.5*stress + 0.01*nodes",
        "return 0.6*hold + 0.3*stress + 0.1*train",
        "return hold + stress + 0.001*nodes",
    ])
    new_stmts.insert(idx, expr)
    return new_stmts


OPERATORS: Dict[str, Callable[[random.Random, List[str]], List[str]]] = {
    "insert_assign": op_insert_assign,
    "insert_if": op_insert_if,
    "insert_while": op_insert_while,
    "delete_stmt": op_delete_stmt,
    "modify_line": op_modify_line,
    "tweak_const": op_tweak_const,
    "change_binary": op_change_binary,
    "list_manip": op_list_manipulation,
    "modify_return": op_modify_return,
    "learner_update": op_learner_update_step,
    "learner_objective": op_learner_objective_tweak,
}
PRIMITIVE_OPS = list(OPERATORS.keys())

# @@OPERATORS_LIB_START@@
OPERATORS_LIB: Dict[str, Dict] = {}
# @@OPERATORS_LIB_END@@


def apply_synthesized_op(rng: random.Random, stmts: List[str], steps: List[str]) -> List[str]:
    result = stmts
    for step in steps:
        if step in OPERATORS:
            result = OPERATORS[step](rng, result)
    return result

def synthesize_new_operator(rng: random.Random) -> Tuple[str, Dict]:
    n_steps = rng.randint(2, 4)
    steps = [rng.choice(PRIMITIVE_OPS) for _ in range(n_steps)]
    name = f"synth_{sha256(''.join(steps) + str(time.time()))[:8]}"
    return (name, {"steps": steps, "score": 0.0})


def mutate_learner(rng: random.Random, learner: LearnerGenome, meta: "MetaState") -> LearnerGenome:
    """PHASE B: mutate a learner genome by selecting a block."""
    blocks = ["encode", "predict", "update", "objective"]
    block = rng.choice(blocks)
    op = meta.sample_op(rng)

    def apply_block(stmts: List[str]) -> List[str]:
        if op in OPERATORS:
            return OPERATORS[op](rng, stmts)
        return stmts

    if block == "encode":
        new_encode = apply_block(learner.encode_stmts)
        return LearnerGenome(new_encode, learner.predict_stmts, learner.update_stmts, learner.objective_stmts, parents=[learner.gid], op_tag=f"mut:{block}:{op}")
    if block == "predict":
        new_predict = apply_block(learner.predict_stmts)
        return LearnerGenome(learner.encode_stmts, new_predict, learner.update_stmts, learner.objective_stmts, parents=[learner.gid], op_tag=f"mut:{block}:{op}")
    if block == "update":
        new_update = apply_block(learner.update_stmts)
        return LearnerGenome(learner.encode_stmts, learner.predict_stmts, new_update, learner.objective_stmts, parents=[learner.gid], op_tag=f"mut:{block}:{op}")
    new_objective = apply_block(learner.objective_stmts)
    return LearnerGenome(learner.encode_stmts, learner.predict_stmts, learner.update_stmts, new_objective, parents=[learner.gid], op_tag=f"mut:{block}:{op}")


# ---------------------------
# Surrogate + MAP-Elites
# ---------------------------

class SurrogateModel:
    def __init__(self, k: int = 5):
        self.k = k
        self.memory: List[Tuple[List[float], float]] = []

    def _extract_features(self, code: str) -> List[float]:
        return [
            len(code),
            code.count("\n"),
            code.count("if "),
            code.count("while "),
            code.count("="),
            code.count("return "),
            code.count("("),
        ]

    def train(self, history: List[Dict]):
        self.memory = []
        for h in history[-200:]:
            src = h.get("code") or h.get("expr")
            if src and "score" in h and isinstance(h["score"], (int, float)):
                feat = self._extract_features(src)
                self.memory.append((feat, float(h["score"])))

    def predict(self, code: str) -> float:
        if not self.memory:
            return 0.0
        target = self._extract_features(code)
        dists = []
        for feat, score in self.memory:
            d = sum((f1 - f2) ** 2 for f1, f2 in zip(target, feat)) ** 0.5
            dists.append((d, score))
        dists.sort(key=lambda x: x[0])
        nearest = dists[: self.k]
        total_w = 0.0
        weighted = 0.0
        for d, s in nearest:
            w = 1.0 / (d + 1e-6)
            weighted += s * w
            total_w += w
        return weighted / total_w if total_w > 0 else 0.0


SURROGATE = SurrogateModel()


class MAPElitesArchive:
    def __init__(self, genome_cls: type = Genome):
        self.grid: Dict[Tuple[int, int], Tuple[float, Any]] = {}
        self.genome_cls = genome_cls

    def _features(self, code: str) -> Tuple[int, int]:
        l_bin = min(20, len(code) // 20)
        d_bin = min(10, code.count("\n") // 2)
        return (l_bin, d_bin)

    def add(self, genome: Any, score: float):
        feat = self._features(genome.code)
        if feat not in self.grid or score < self.grid[feat][0]:
            self.grid[feat] = (score, genome)

    def sample(self, rng: random.Random) -> Optional[Any]:
        if not self.grid:
            return None
        return rng.choice(list(self.grid.values()))[1]

    def snapshot(self) -> Dict:
        return {
            "grid_size": len(self.grid),
            "entries": [(list(k), v[0], asdict(v[1])) for k, v in self.grid.items()],
        }

    def from_snapshot(self, s: Dict) -> "MAPElitesArchive":
        ma = MAPElitesArchive(self.genome_cls)
        for k, score, g_dict in s.get("entries", []):
            ma.grid[tuple(k)] = (score, self.genome_cls(**g_dict))
        return ma


MAP_ELITES = MAPElitesArchive(Genome)
MAP_ELITES_LEARNER = MAPElitesArchive(LearnerGenome)

def map_elites_filename(mode: str) -> str:
    return "map_elites_learner.json" if mode == "learner" else "map_elites.json"

def save_map_elites(path: Path, archive: MAPElitesArchive):
    path.write_text(json.dumps(archive.snapshot(), indent=2), encoding="utf-8")

def load_map_elites(path: Path, archive: MAPElitesArchive):
    if path.exists():
        try:
            loaded = archive.from_snapshot(json.loads(path.read_text(encoding="utf-8")))
            archive.grid = loaded.grid
        except Exception:
            pass


# ---------------------------
# Operator library evolution
# ---------------------------

def evolve_operator_meta(rng: random.Random) -> Tuple[str, Dict]:
    candidates = [v for _, v in OPERATORS_LIB.items() if v.get("score", 0) > -5.0]
    if len(candidates) < 2:
        return synthesize_new_operator(rng)
    p1 = rng.choice(candidates)["steps"]
    p2 = rng.choice(candidates)["steps"]
    cut = rng.randint(0, min(len(p1), len(p2)))
    child_steps = p1[:cut] + p2[cut:]
    if rng.random() < 0.5:
        mut_type = rng.choice(["mod", "add", "del"])
        if mut_type == "mod" and child_steps:
            child_steps[rng.randint(0, len(child_steps) - 1)] = rng.choice(PRIMITIVE_OPS)
        elif mut_type == "add":
            child_steps.insert(rng.randint(0, len(child_steps)), rng.choice(PRIMITIVE_OPS))
        elif mut_type == "del" and len(child_steps) > 1:
            child_steps.pop(rng.randint(0, len(child_steps) - 1))
    child_steps = child_steps[:6] or [rng.choice(PRIMITIVE_OPS)]
    name = f"evo_{sha256(''.join(child_steps) + str(time.time()))[:8]}"
    return (name, {"steps": child_steps, "score": 0.0})

def maybe_evolve_operators_lib(rng: random.Random, threshold: int = 10) -> Optional[str]:
    # remove worst if very bad
    if len(OPERATORS_LIB) > 3:
        sorted_ops = sorted(OPERATORS_LIB.items(), key=lambda x: x[1].get("score", 0))
        worst_name, worst_spec = sorted_ops[0]
        if worst_spec.get("score", 0) < -threshold:
            del OPERATORS_LIB[worst_name]

    # add new until size
    if len(OPERATORS_LIB) < 8:
        if rng.random() < 0.7 and len(OPERATORS_LIB) >= 2:
            name, spec = evolve_operator_meta(rng)
        else:
            name, spec = synthesize_new_operator(rng)
        OPERATORS_LIB[name] = spec
        return name
    return None


# ---------------------------
# Curriculum generator (simple)
# ---------------------------

class ProblemGenerator:
    def __init__(self):
        self.archive: List[Dict] = []

    def evolve_task(self, rng: random.Random, current_elites: List[Genome]) -> TaskSpec:
        arc_tasks = get_arc_tasks()
        base_options = ["sort", "reverse", "max", "filter"]
        arc_options = [f"arc_{tid}" for tid in arc_tasks] if arc_tasks else []
        options = base_options + arc_options
        base_name = rng.choice(options) if options else "sort"
        level = rng.randint(1, 3)
        mn = 3 + level
        mx = 5 + level
        if base_name.startswith("arc_"):
            mn, mx = (3, 5)
        return TaskSpec(name=base_name, n_train=64, n_hold=32, x_min=float(mn), x_max=float(mx), noise=0.0)


# ---------------------------
# Task detective (seeding hints)
# ---------------------------

class TaskDetective:
    @staticmethod
    def detect_pattern(batch: Optional[Batch]) -> Optional[str]:
        if not batch or not batch.x_tr:
            return None
        check_set = list(zip(batch.x_tr[:5], batch.y_tr[:5]))
        is_sort = is_rev = is_max = is_min = is_len = True
        for x, y in check_set:
            if not isinstance(x, list) or not isinstance(y, (list, int, float)):
                return None
            if isinstance(y, list):
                if y != sorted(x):
                    is_sort = False
                if y != list(reversed(x)):
                    is_rev = False
            else:
                is_sort = is_rev = False
            if isinstance(y, (int, float)):
                if not x:
                    if y != 0:
                        is_len = False
                else:
                    if y != len(x):
                        is_len = False
                    if y != max(x):
                        is_max = False
                    if y != min(x):
                        is_min = False
            else:
                is_max = is_min = is_len = False
        if is_sort:
            return "HINT_SORT"
        if is_rev:
            return "HINT_REVERSE"
        if is_max:
            return "HINT_MAX"
        if is_min:
            return "HINT_MIN"
        if is_len:
            return "HINT_LEN"
        return None


def seed_genome(rng: random.Random, hint: Optional[str] = None) -> Genome:
    seeds = [
        ["return x"],
        ["return sorted(x)"],
        ["return list(reversed(x))"],
        ["v0 = sorted(x)", "return v0"],
        [f"return {_random_expr(rng, depth=0)}"],
    ]
    if hint == "HINT_SORT":
        seeds.extend([["return sorted(x)"]] * 5)
    elif hint == "HINT_REVERSE":
        seeds.extend([["return list(reversed(x))"]] * 5)
    elif hint == "HINT_MAX":
        seeds.extend([["return max(x)"]] * 5)
    elif hint == "HINT_MIN":
        seeds.extend([["return min(x)"]] * 5)
    elif hint == "HINT_LEN":
        seeds.extend([["return len(x)"]] * 5)
    return Genome(statements=rng.choice(seeds))


def seed_learner_genome(rng: random.Random, hint: Optional[str] = None) -> LearnerGenome:
    """PHASE B: learner seed set with simple predictors and objectives."""
    base_encode = ["return x"]
    base_predict = ["return z"]
    base_update = ["return mem"]
    base_obj = ["return hold + 0.5*stress + 0.01*nodes"]

    linear_predict = ["return mem['w'] * z + mem['b']"]
    linear_update = [
        "mem['w'] = mem['w'] + lr * (y_true - y_pred) * z",
        "mem['b'] = mem['b'] + lr * (y_true - y_pred)",
        "return mem",
    ]

    list_sort_predict = ["return sorted(z)"]
    list_reverse_predict = ["return list(reversed(z))"]
    list_max_predict = ["return max(z) if z else 0"]

    seeds = [
        LearnerGenome(base_encode, base_predict, base_update, base_obj),
        LearnerGenome(base_encode, linear_predict, linear_update, base_obj),
    ]

    if hint == "HINT_SORT":
        seeds.append(LearnerGenome(base_encode, list_sort_predict, base_update, base_obj))
    elif hint == "HINT_REVERSE":
        seeds.append(LearnerGenome(base_encode, list_reverse_predict, base_update, base_obj))
    elif hint == "HINT_MAX":
        seeds.append(LearnerGenome(base_encode, list_max_predict, base_update, base_obj))

    return rng.choice(seeds)


# ---------------------------
# Function library (learned helpers)
# ---------------------------

@dataclass
class LearnedFunc:
    name: str
    expr: str
    trust: float = 1.0
    uses: int = 0


class FunctionLibrary:
    def __init__(self, max_size: int = 16):
        self.funcs: Dict[str, LearnedFunc] = {}
        self.max_size = max_size

    def maybe_adopt(self, rng: random.Random, expr: str, threshold: float = 0.1) -> Optional[str]:
        if len(self.funcs) >= self.max_size or rng.random() > threshold:
            return None
        try:
            tree = ast.parse(expr, mode="eval").body
            nodes = list(ast.walk(tree))
            if len(nodes) < 4:
                return None
            sub = _pick_node(rng, tree)
            sub_expr = _to_src(sub)
            if node_count(sub_expr) < 3:
                return None
            ok, _ = validate_expr(sub_expr, extra=set(self.funcs.keys()))
            if not ok:
                return None
            name = f"h{len(self.funcs) + 1}"
            self.funcs[name] = LearnedFunc(name=name, expr=sub_expr)
            return name
        except Exception:
            return None

    def maybe_inject(self, rng: random.Random, expr: str) -> Tuple[str, Optional[str]]:
        if not self.funcs or rng.random() > 0.2:
            return (expr, None)
        fn = rng.choice(list(self.funcs.values()))
        fn.uses += 1
        try:
            call = f"{fn.name}(x)"
            new = expr.replace("x", call, 1) if rng.random() < 0.5 else f"({expr}+{call})"
            ok, _ = validate_expr(new, extra=set(self.funcs.keys()))
            return (new, fn.name) if ok else (expr, None)
        except Exception:
            return (expr, None)

    def update_trust(self, name: str, improved: bool):
        if name in self.funcs:
            self.funcs[name].trust *= 1.1 if improved else 0.9
            self.funcs[name].trust = clamp(self.funcs[name].trust, 0.1, 10.0)

    def get_helpers(self) -> Dict[str, Callable]:
        # helper functions callable from evolved programs
        helpers: Dict[str, Callable] = {}

        def make_helper(expr: str):
            return lambda x: safe_eval(expr, x, extra_funcs=helpers)

        for n, f in self.funcs.items():
            helpers[n] = make_helper(f.expr)
        return helpers

    def snapshot(self) -> Dict:
        return {"funcs": [asdict(f) for f in self.funcs.values()]}

    def merge(self, other: "FunctionLibrary"):
        for name, func in other.funcs.items():
            if name not in self.funcs:
                self.funcs[name] = func
            else:
                new_name = f"{name}_{len(self.funcs) + 1}"
                self.funcs[new_name] = LearnedFunc(name=new_name, expr=func.expr, trust=func.trust, uses=func.uses)

    @staticmethod
    def from_snapshot(s: Dict) -> "FunctionLibrary":
        lib = FunctionLibrary()
        for fd in s.get("funcs", []):
            lib.funcs[fd["name"]] = LearnedFunc(**fd)
        return lib


@dataclass
class LibraryRecord:
    descriptor: TaskDescriptor
    score_hold: float
    snapshot: Dict[str, Any]


class LibraryArchive:
    def __init__(self, k: int = 2):
        self.k = k
        self.records: List[LibraryRecord] = []

    def add(self, descriptor: TaskDescriptor, score_hold: float, lib: FunctionLibrary):
        self.records.append(LibraryRecord(descriptor=descriptor, score_hold=score_hold, snapshot=lib.snapshot()))

    def _distance(self, a: List[float], b: List[float]) -> float:
        return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))

    def select(self, descriptor: TaskDescriptor) -> List[FunctionLibrary]:
        if not self.records:
            return []
        vec = descriptor.vector()
        ranked = sorted(self.records, key=lambda r: (self._distance(vec, r.descriptor.vector()), r.score_hold))
        libs = []
        for rec in ranked[: self.k]:
            libs.append(FunctionLibrary.from_snapshot(rec.snapshot))
        return libs


# ---------------------------
# Grammar induction (single definition)
# ---------------------------

def induce_grammar(pool: List[Genome]):
    if not pool:
        return
    elites = pool[: max(10, len(pool) // 5)]
    counts = {k: 0.1 for k in GRAMMAR_PROBS}
    for g in elites:
        try:
            tree = ast.parse(g.code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id in counts:
                        counts[node.func.id] += 1.0
                    counts["call"] += 1.0
                elif isinstance(node, ast.BinOp):
                    counts["binop"] += 1.0
                elif isinstance(node, ast.Name) and node.id == "x":
                    counts["var"] += 1.0
                elif isinstance(node, ast.Constant):
                    counts["const"] += 1.0
        except Exception:
            pass
    total = sum(counts.values())
    if total > 0:
        for k in counts:
            old = GRAMMAR_PROBS.get(k, 1.0)
            target = counts[k] / total * 100.0
            GRAMMAR_PROBS[k] = 0.8 * old + 0.2 * target


def extract_return_expr(stmts: List[str]) -> Optional[str]:
    for stmt in reversed(stmts):
        s = stmt.strip()
        if s.startswith("return "):
            return s[len("return ") :].strip()
    return None


def inject_helpers_into_statements(rng: random.Random, stmts: List[str], library: FunctionLibrary) -> List[str]:
    if not library.funcs:
        return stmts
    new_stmts = []
    injected = False
    for stmt in stmts:
        if not injected and stmt.strip().startswith("return "):
            expr = stmt.strip()[len("return ") :].strip()
            new_expr, helper_name = library.maybe_inject(rng, expr)
            if helper_name:
                stmt = f"return {new_expr}"
                injected = True
        new_stmts.append(stmt)
    return new_stmts


# ---------------------------
# MetaState (L0/L1 source-patchable)
# ---------------------------

OP_WEIGHT_INIT: Dict[str, float] = {
    k: (5.0 if k in ("modify_return", "insert_assign", "list_manip") else 1.0)
    for k in OPERATORS
}


@dataclass
class UpdateRuleGenome:
    """Representation for update rule parameters (meta-level learning algorithm)."""

    learning_rate: float
    momentum: float
    rejection_penalty: float
    reward_scale: float
    uid: str = ""

    def __post_init__(self):
        if not self.uid:
            self.uid = sha256(f"{self.learning_rate}:{self.momentum}:{self.rejection_penalty}:{self.reward_scale}")[:10]

    def apply(self, meta: "MetaState", op: str, delta: float, accepted: bool) -> None:
        reward = (max(0.0, -delta) * self.reward_scale) if accepted else -self.rejection_penalty
        velocity = meta.op_velocity.get(op, 0.0)
        velocity = self.momentum * velocity + reward
        meta.op_velocity[op] = velocity
        meta.op_weights[op] = clamp(meta.op_weights.get(op, 1.0) + self.learning_rate * velocity, 0.1, 8.0)

    def mutate(self, rng: random.Random) -> "UpdateRuleGenome":
        return UpdateRuleGenome(
            learning_rate=clamp(self.learning_rate + rng.uniform(-0.05, 0.05), 0.01, 0.5),
            momentum=clamp(self.momentum + rng.uniform(-0.1, 0.1), 0.0, 0.95),
            rejection_penalty=clamp(self.rejection_penalty + rng.uniform(-0.05, 0.05), 0.01, 0.5),
            reward_scale=clamp(self.reward_scale + rng.uniform(-0.1, 0.1), 0.2, 2.0),
        )

    @staticmethod
    def default() -> "UpdateRuleGenome":
        return UpdateRuleGenome(learning_rate=0.12, momentum=0.3, rejection_penalty=0.12, reward_scale=1.0)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "UpdateRuleGenome":
        return UpdateRuleGenome(
            learning_rate=float(data.get("learning_rate", 0.12)),
            momentum=float(data.get("momentum", 0.3)),
            rejection_penalty=float(data.get("rejection_penalty", 0.12)),
            reward_scale=float(data.get("reward_scale", 1.0)),
            uid=data.get("uid", ""),
        )

@dataclass
class MetaState:
    op_weights: Dict[str, float] = field(default_factory=lambda: dict(OP_WEIGHT_INIT))
    op_velocity: Dict[str, float] = field(default_factory=dict)
    mutation_rate: float = 0.8863
    crossover_rate: float = 0.1971
    complexity_lambda: float = 0.0001
    epsilon_explore: float = 0.4213
    adapt_steps: int = 8
    stuck_counter: int = 0
    update_rule: UpdateRuleGenome = field(default_factory=UpdateRuleGenome.default)
    strategy: EngineStrategy = field(default_factory=lambda: EngineStrategy(
        selection_code=DEFAULT_SELECTION_CODE,
        crossover_code=DEFAULT_CROSSOVER_CODE,
        mutation_policy_code=DEFAULT_MUTATION_CODE
    ))

    def sample_op(self, rng: random.Random) -> str:
        if rng.random() < self.epsilon_explore:
            return rng.choice(list(OPERATORS.keys()))
        total = sum(max(0.01, w) for w in self.op_weights.values())
        r = rng.random() * total
        acc = 0.0
        for k, w in self.op_weights.items():
            acc += max(0.01, w)
            if r <= acc:
                return k
        return rng.choice(list(OPERATORS.keys()))

    def update(self, op: str, delta: float, accepted: bool):
        if op in self.op_weights:
            self.update_rule.apply(self, op, delta, accepted)
        if not accepted:
            self.stuck_counter += 1
            if self.stuck_counter > 20:
                self.epsilon_explore = clamp(self.epsilon_explore + 0.02, 0.1, 0.4)
                self.mutation_rate = clamp(self.mutation_rate + 0.03, 0.4, 0.95)
        else:
            self.stuck_counter = 0
            self.epsilon_explore = clamp(self.epsilon_explore - 0.01, 0.05, 0.3)


class MetaCognitiveEngine:
    @staticmethod
    def analyze_execution(results: List[Tuple[Any, EvalResult]], meta: MetaState):
        errors = [r.err.split(":")[0] for _, r in results if (not r.ok and r.err)]
        if not errors:
            return
        counts = collections.Counter(errors)
        total_err = len(errors)
        if counts.get("TypeError", 0) > total_err * 0.3:
            if "binop" in GRAMMAR_PROBS:
                GRAMMAR_PROBS["binop"] *= 0.5
            GRAMMAR_PROBS["var"] = GRAMMAR_PROBS.get("var", 1.0) * 1.5
        if counts.get("IndexError", 0) > total_err * 0.3:
            if "list_manip" in meta.op_weights:
                meta.op_weights["list_manip"] *= 0.7
        if counts.get("StepLimitExceeded", 0) > total_err * 0.3:
            meta.complexity_lambda *= 2.0


# ---------------------------
# L1 Meta-optimizer policy
# ---------------------------

@dataclass
class MetaPolicy:
    weights: List[List[float]]
    bias: List[float]
    pid: str = ""

    @staticmethod
    def seed(rng: random.Random, n_outputs: int, n_inputs: int) -> "MetaPolicy":
        weights = [[rng.uniform(-0.2, 0.2) for _ in range(n_inputs)] for _ in range(n_outputs)]
        bias = [rng.uniform(-0.1, 0.1) for _ in range(n_outputs)]
        pid = sha256(json.dumps(weights) + json.dumps(bias))[:10]
        return MetaPolicy(weights=weights, bias=bias, pid=pid)

    def _linear(self, features: List[float], idx: int) -> float:
        w = self.weights[idx]
        return sum(fi * wi for fi, wi in zip(features, w)) + self.bias[idx]

    def act(self, descriptor: TaskDescriptor, stats: Dict[str, float]) -> Dict[str, Any]:
        features = descriptor.vector() + [
            stats.get("delta_best", 0.0),
            stats.get("auc_window", 0.0),
            stats.get("timeout_rate", 0.0),
            stats.get("avg_nodes", 0.0),
        ]
        outputs = [self._linear(features, i) for i in range(len(self.weights))]
        mutation_rate = clamp(0.5 + outputs[0], 0.05, 0.98)
        crossover_rate = clamp(0.2 + outputs[1], 0.0, 0.9)
        novelty_weight = clamp(0.2 + outputs[2], 0.0, 1.0)
        branch_insert_rate = clamp(0.1 + outputs[3], 0.0, 0.6)
        op_scale = clamp(1.0 + outputs[4], 0.2, 3.0)
        op_weights = {
            "modify_return": clamp(OP_WEIGHT_INIT.get("modify_return", 1.0) * op_scale, 0.1, 8.0),
            "insert_assign": clamp(OP_WEIGHT_INIT.get("insert_assign", 1.0) * (op_scale + 0.2), 0.1, 8.0),
            "list_manip": clamp(OP_WEIGHT_INIT.get("list_manip", 1.0) * (op_scale - 0.1), 0.1, 8.0),
        }
        return {
            "mutation_rate": mutation_rate,
            "crossover_rate": crossover_rate,
            "novelty_weight": novelty_weight,
            "branch_insert_rate": branch_insert_rate,
            "op_weights": op_weights,
        }

    def mutate(self, rng: random.Random, scale: float = 0.1) -> "MetaPolicy":
        weights = [row[:] for row in self.weights]
        bias = self.bias[:]
        for i in range(len(weights)):
            if rng.random() < 0.7:
                j = rng.randrange(len(weights[i]))
                weights[i][j] += rng.uniform(-scale, scale)
        for i in range(len(bias)):
            if rng.random() < 0.5:
                bias[i] += rng.uniform(-scale, scale)
        pid = sha256(json.dumps(weights) + json.dumps(bias))[:10]
        return MetaPolicy(weights=weights, bias=bias, pid=pid)


# ---------------------------
# Duo-loop (Creator/Critic)
# ---------------------------

@dataclass
class AgentPolicy:
    generator_mode: str
    search_bias: Dict[str, float]
    gate_target: float
    slice_seconds: float


CREATOR_POLICY = AgentPolicy(
    generator_mode="synthesize",
    search_bias={
        "novelty": 1.2,
        "simplicity": 0.4,
        "robustness": 0.3,
        "generalization": 0.2,
        "perf": 0.2,
    },
    gate_target=0.35,
    slice_seconds=6.0,
)

CRITIC_POLICY = AgentPolicy(
    generator_mode="mutate",
    search_bias={
        "novelty": 0.2,
        "simplicity": 0.9,
        "robustness": 1.1,
        "generalization": 1.0,
        "perf": 0.6,
    },
    gate_target=0.7,
    slice_seconds=6.0,
)

# ---------------------------
# Universe / Multiverse
# ---------------------------

@dataclass
class Universe:
    uid: int
    seed: int
    meta: MetaState
    pool: List[Genome]
    library: FunctionLibrary
    discriminator: ProblemGenerator = field(default_factory=ProblemGenerator)
    eval_mode: str = "solver"
    best: Optional[Genome] = None
    best_score: float = float("inf")
    best_train: float = float("inf")
    best_hold: float = float("inf")
    best_stress: float = float("inf")
    best_test: float = float("inf")
    history: List[Dict] = field(default_factory=list)

    def step(
        self,
        gen: int,
        task: TaskSpec,
        pop_size: int,
        batch: Batch,
        policy_controls: Optional[Union[Dict[str, float], ControlPacket]] = None,
    ) -> Dict:
        rng = random.Random(self.seed + gen * 1009)
        if batch is None:
            self.pool = [seed_genome(rng) for _ in range(pop_size)]
            return {"gen": gen, "accepted": False, "reason": "no_batch"}

        helper_env = self.library.get_helpers()
        if policy_controls:
            self.meta.mutation_rate = clamp(policy_controls.get("mutation_rate", self.meta.mutation_rate), 0.05, 0.98)
            self.meta.crossover_rate = clamp(policy_controls.get("crossover_rate", self.meta.crossover_rate), 0.0, 0.95)
            novelty_weight = clamp(policy_controls.get("novelty_weight", 0.0), 0.0, 1.0)
            branch_rate = clamp(policy_controls.get("branch_insert_rate", 0.0), 0.0, 0.6)
            if isinstance(policy_controls.get("op_weights"), dict):
                for k, v in policy_controls["op_weights"].items():
                    if k in self.meta.op_weights:
                        self.meta.op_weights[k] = clamp(float(v), 0.1, 8.0)
        else:
            novelty_weight = 0.0
            branch_rate = 0.0

        scored: List[Tuple[Genome, EvalResult]] = []
        all_results: List[Tuple[Genome, EvalResult]] = []
        for g in self.pool:
            # Hard gate: enforce input dependence before any scoring/selection.
            gate_ok, gate_reason = _hard_gate_ok(
                g.code,
                batch,
                self.eval_mode if self.eval_mode != "program" else "solver",
                task.name,
                extra_env=helper_env,
            )
            if not gate_ok:
                res = EvalResult(
                    False,
                    float("inf"),
                    float("inf"),
                    float("inf"),
                    float("inf"),
                    node_count(g.code),
                    float("inf"),
                    f"hard_gate:{gate_reason}",
                )
                all_results.append((g, res))
                continue
            if self.eval_mode == "algo":
                res = evaluate_algo(g, batch, task.name, self.meta.complexity_lambda)
            else:
                validator = validate_program if self.eval_mode == "program" else validate_code
                res = evaluate(g, batch, task.name, self.meta.complexity_lambda, extra_env=helper_env, validator=validator)
            all_results.append((g, res))
            if res.ok:
                scored.append((g, res))

        MetaCognitiveEngine.analyze_execution(all_results, self.meta)

        if not scored:
            hint = TaskDetective.detect_pattern(batch)
            self.pool = [seed_genome(rng, hint) for _ in range(pop_size)]
            return {"gen": gen, "accepted": False, "reason": "reseed"}

        scored.sort(key=lambda t: t[1].score)
        timeout_rate = 1.0 - (len(scored) / max(1, len(all_results)))
        avg_nodes = sum(r.nodes for _, r in scored) / max(1, len(scored))

        # MAP-Elites add
        best_g0, best_res0 = scored[0]
        MAP_ELITES.add(best_g0, best_res0.score)

        for g, _ in scored[:3]:
            expr = extract_return_expr(g.statements)
            if expr:
                adopted = self.library.maybe_adopt(rng, expr, threshold=0.3)
                if adopted:
                    break

        # selection via strategy
        sel_ctx = {
            "pool": [g for g, _ in scored],
            "scores": [res.score for _, res in scored],
            "pop_size": pop_size,
            "map_elites": MAP_ELITES,
            "rng": rng,
        }
        sel_res = safe_exec_engine(self.meta.strategy.selection_code, sel_ctx)
        if sel_res and isinstance(sel_res, (tuple, list)) and len(sel_res) == 2:
            elites, parenting_pool = sel_res
        else:
            elites = [g for g, _ in scored[: max(4, pop_size // 10)]]
            parenting_pool = [rng.choice(elites) for _ in range(pop_size - len(elites))]

        candidates: List[Genome] = []
        needed = pop_size - len(elites)
        attempts_needed = max(needed * 2, needed + 8)
        mate_pool = list(elites) + list(parenting_pool)

        while len(candidates) < attempts_needed:
            parent = rng.choice(parenting_pool) if parenting_pool else rng.choice(elites)
            new_stmts = None
            op_tag = "copy"

            # crossover
            if rng.random() < self.meta.crossover_rate and len(mate_pool) > 1:
                p2 = rng.choice(mate_pool)
                cross_ctx = {"p1": parent.statements, "p2": p2.statements, "rng": rng}
                new_stmts = safe_exec_engine(self.meta.strategy.crossover_code, cross_ctx)
                if new_stmts and isinstance(new_stmts, list):
                    op_tag = "crossover"
                else:
                    new_stmts = None

            if not new_stmts:
                new_stmts = parent.statements[:]

            # mutation
            if op_tag in ("copy", "crossover") and rng.random() < self.meta.mutation_rate:
                use_synth = rng.random() < 0.3 and bool(OPERATORS_LIB)
                if use_synth:
                    synth_name = rng.choice(list(OPERATORS_LIB.keys()))
                    steps = OPERATORS_LIB[synth_name].get("steps", [])
                    new_stmts = apply_synthesized_op(rng, new_stmts, steps)
                    op_tag = f"synth:{synth_name}"
                else:
                    op = self.meta.sample_op(rng)
                    if op in OPERATORS:
                        new_stmts = OPERATORS[op](rng, new_stmts)
                    op_tag = f"mut:{op}"

            if rng.random() < branch_rate:
                extra = rng.choice(seed_genome(rng).statements)
                new_stmts = list(new_stmts) + [extra]
                op_tag = f"{op_tag}|branch"

            new_stmts = inject_helpers_into_statements(rng, list(new_stmts), self.library)
            candidates.append(Genome(statements=new_stmts, parents=[parent.gid], op_tag=op_tag))

        # surrogate ranking
        with_pred = [(c, SURROGATE.predict(c.code) + novelty_weight * rng.random()) for c in candidates]
        with_pred.sort(key=lambda x: x[1])
        selected_children = [c for c, _ in with_pred[:needed]]

        self.pool = list(elites) + selected_children

        # occasionally evolve operator library
        if rng.random() < 0.02:
            maybe_evolve_operators_lib(rng)

        # grammar induction
        if gen % 5 == 0:
            induce_grammar(list(elites))

        # acceptance update
        best_g, best_res = scored[0]
        old_score = self.best_score
        accept_margin = 1e-9
        if isinstance(policy_controls, ControlPacket):
            accept_margin = max(accept_margin, policy_controls.acceptance_margin)
        accepted = best_res.score < self.best_score - accept_margin
        if accepted:
            self.best = best_g
            self.best_score = best_res.score
            self.best_train = best_res.train
            self.best_hold = best_res.hold
            self.best_stress = best_res.stress
            self.best_test = best_res.test

        op_used = best_g.op_tag.split(":")[1].split("|")[0] if ":" in best_g.op_tag else "unknown"
        self.meta.update(op_used, self.best_score - old_score, accepted)
        if isinstance(policy_controls, ControlPacket) and self.meta.stuck_counter > policy_controls.patience:
            self.meta.epsilon_explore = clamp(self.meta.epsilon_explore + 0.05, 0.05, 0.5)

        log = {
            "gen": gen,
            "accepted": accepted,
            "score": self.best_score,
            "train": self.best_train,
            "hold": self.best_hold,
            "stress": self.best_stress,
            "test": self.best_test,
            "code": self.best.code if self.best else "none",
            "novelty_weight": novelty_weight,
            "timeout_rate": timeout_rate,
            "avg_nodes": avg_nodes,
        }
        self.history.append(log)
        if gen % 5 == 0:
            SURROGATE.train(self.history)
        return log

    def snapshot(self) -> Dict:
        return {
            "uid": self.uid,
            "seed": self.seed,
            "meta": asdict(self.meta),
            "best": asdict(self.best) if self.best else None,
            "best_score": self.best_score,
            "best_train": self.best_train,
            "best_hold": self.best_hold,
            "best_stress": self.best_stress,
            "best_test": self.best_test,
            "pool": [asdict(g) for g in self.pool[:20]],
            "library": self.library.snapshot(),
            "history": self.history[-50:],
            "eval_mode": self.eval_mode,
        }

    @staticmethod
    def from_snapshot(s: Dict) -> "Universe":
        meta_data = s.get("meta", {})
        if "strategy" in meta_data and isinstance(meta_data["strategy"], dict):
            meta_data["strategy"] = EngineStrategy(**meta_data["strategy"])
        if "update_rule" in meta_data and isinstance(meta_data["update_rule"], dict):
            meta_data["update_rule"] = UpdateRuleGenome.from_dict(meta_data["update_rule"])
        meta = MetaState(**{k: v for k, v in meta_data.items() if k != "op_weights"})
        meta.op_weights = meta_data.get("op_weights", dict(OP_WEIGHT_INIT))
        pool = [Genome(**g) for g in s.get("pool", [])]
        lib = FunctionLibrary.from_snapshot(s.get("library", {}))
        u = Universe(uid=s.get("uid", 0), seed=s.get("seed", 0), meta=meta, pool=pool, library=lib)
        if s.get("best"):
            u.best = Genome(**s["best"])
        u.best_score = s.get("best_score", float("inf"))
        u.best_train = s.get("best_train", float("inf"))
        u.best_hold = s.get("best_hold", float("inf"))
        u.best_stress = s.get("best_stress", float("inf"))
        u.best_test = s.get("best_test", float("inf"))
        u.history = s.get("history", [])
        u.eval_mode = s.get("eval_mode", "solver")
        return u


@dataclass
class UniverseLearner:
    """PHASE C: learner multiverse wrapper."""
    uid: int
    seed: int
    meta: MetaState
    pool: List[LearnerGenome]
    library: FunctionLibrary
    discriminator: ProblemGenerator = field(default_factory=ProblemGenerator)
    best: Optional[LearnerGenome] = None
    best_score: float = float("inf")
    best_hold: float = float("inf")
    best_stress: float = float("inf")
    best_test: float = float("inf")
    history: List[Dict] = field(default_factory=list)

    def step(self, gen: int, task: TaskSpec, pop_size: int, batch: Batch) -> Dict:
        rng = random.Random(self.seed + gen * 1009)
        if batch is None:
            hint = TaskDetective.detect_pattern(batch)
            self.pool = [seed_learner_genome(rng, hint) for _ in range(pop_size)]
            return {"gen": gen, "accepted": False, "reason": "no_batch"}

        scored: List[Tuple[LearnerGenome, EvalResult]] = []
        all_results: List[Tuple[LearnerGenome, EvalResult]] = []
        for g in self.pool:
            # Hard gate: enforce input dependence before any scoring/selection.
            gate_ok, gate_reason = _hard_gate_ok(g.code, batch, "learner", task.name)
            if not gate_ok:
                res = EvalResult(
                    False,
                    float("inf"),
                    float("inf"),
                    float("inf"),
                    float("inf"),
                    node_count(g.code),
                    float("inf"),
                    f"hard_gate:{gate_reason}",
                )
                all_results.append((g, res))
                continue
            res = evaluate_learner(g, batch, task.name, self.meta.adapt_steps, self.meta.complexity_lambda)
            all_results.append((g, res))
            if res.ok:
                scored.append((g, res))

        MetaCognitiveEngine.analyze_execution(all_results, self.meta)

        if not scored:
            hint = TaskDetective.detect_pattern(batch)
            self.pool = [seed_learner_genome(rng, hint) for _ in range(pop_size)]
            return {"gen": gen, "accepted": False, "reason": "reseed"}

        scored.sort(key=lambda t: t[1].score)
        best_g0, best_res0 = scored[0]
        MAP_ELITES_LEARNER.add(best_g0, best_res0.score)

        sel_ctx = {
            "pool": [g for g, _ in scored],
            "scores": [res.score for _, res in scored],
            "pop_size": pop_size,
            "map_elites": MAP_ELITES_LEARNER,
            "rng": rng,
        }
        sel_res = safe_exec_engine(self.meta.strategy.selection_code, sel_ctx)
        if sel_res and isinstance(sel_res, (tuple, list)) and len(sel_res) == 2:
            elites, parenting_pool = sel_res
        else:
            elites = [g for g, _ in scored[: max(4, pop_size // 10)]]
            parenting_pool = [rng.choice(elites) for _ in range(pop_size - len(elites))]

        candidates: List[LearnerGenome] = []
        needed = pop_size - len(elites)
        attempts_needed = max(needed * 2, needed + 8)
        mate_pool = list(elites) + list(parenting_pool)

        while len(candidates) < attempts_needed:
            parent = rng.choice(parenting_pool) if parenting_pool else rng.choice(elites)
            child = parent
            op_tag = "copy"

            if rng.random() < self.meta.crossover_rate and len(mate_pool) > 1:
                p2 = rng.choice(mate_pool)
                new_encode = safe_exec_engine(self.meta.strategy.crossover_code, {"p1": parent.encode_stmts, "p2": p2.encode_stmts, "rng": rng})
                new_predict = safe_exec_engine(self.meta.strategy.crossover_code, {"p1": parent.predict_stmts, "p2": p2.predict_stmts, "rng": rng})
                new_update = safe_exec_engine(self.meta.strategy.crossover_code, {"p1": parent.update_stmts, "p2": p2.update_stmts, "rng": rng})
                new_objective = safe_exec_engine(self.meta.strategy.crossover_code, {"p1": parent.objective_stmts, "p2": p2.objective_stmts, "rng": rng})
                if all(isinstance(v, list) for v in (new_encode, new_predict, new_update, new_objective)):
                    child = LearnerGenome(new_encode, new_predict, new_update, new_objective, parents=[parent.gid], op_tag="crossover")
                    op_tag = "crossover"

            if op_tag in ("copy", "crossover") and rng.random() < self.meta.mutation_rate:
                child = mutate_learner(rng, child, self.meta)
                op_tag = child.op_tag

            candidates.append(child)

        with_pred = [(c, SURROGATE.predict(c.code)) for c in candidates]
        with_pred.sort(key=lambda x: x[1])
        selected_children = [c for c, _ in with_pred[:needed]]

        self.pool = list(elites) + selected_children

        if rng.random() < 0.02:
            maybe_evolve_operators_lib(rng)

        if gen % 5 == 0:
            induce_grammar([Genome(statements=["return x"])])

        best_g, best_res = scored[0]
        old_score = self.best_score
        accepted = best_res.score < self.best_score - 1e-9
        if accepted:
            self.best = best_g
            self.best_score = best_res.score
            self.best_hold = best_res.hold
            self.best_stress = best_res.stress
            self.best_test = best_res.test

        op_used = best_g.op_tag.split(":")[1].split("|")[0] if ":" in best_g.op_tag else "unknown"
        self.meta.update(op_used, self.best_score - old_score, accepted)

        log = {
            "gen": gen,
            "accepted": accepted,
            "score": self.best_score,
            "hold": self.best_hold,
            "stress": self.best_stress,
            "test": self.best_test,
            "code": self.best.code if self.best else "none",
        }
        self.history.append(log)
        if gen % 5 == 0:
            SURROGATE.train(self.history)
        return log

    def snapshot(self) -> Dict:
        return {
            "uid": self.uid,
            "seed": self.seed,
            "meta": asdict(self.meta),
            "best": asdict(self.best) if self.best else None,
            "best_score": self.best_score,
            "best_hold": self.best_hold,
            "best_stress": self.best_stress,
            "best_test": self.best_test,
            "pool": [asdict(g) for g in self.pool[:20]],
            "library": self.library.snapshot(),
            "history": self.history[-50:],
        }

    @staticmethod
    def from_snapshot(s: Dict) -> "UniverseLearner":
        meta_data = s.get("meta", {})
        if "strategy" in meta_data and isinstance(meta_data["strategy"], dict):
            meta_data["strategy"] = EngineStrategy(**meta_data["strategy"])
        if "update_rule" in meta_data and isinstance(meta_data["update_rule"], dict):
            meta_data["update_rule"] = UpdateRuleGenome.from_dict(meta_data["update_rule"])
        meta = MetaState(**{k: v for k, v in meta_data.items() if k != "op_weights"})
        meta.op_weights = meta_data.get("op_weights", dict(OP_WEIGHT_INIT))
        pool = [LearnerGenome(**g) for g in s.get("pool", [])]
        lib = FunctionLibrary.from_snapshot(s.get("library", {}))
        u = UniverseLearner(uid=s.get("uid", 0), seed=s.get("seed", 0), meta=meta, pool=pool, library=lib)
        if s.get("best"):
            u.best = LearnerGenome(**s["best"])
        u.best_score = s.get("best_score", float("inf"))
        u.best_hold = s.get("best_hold", float("inf"))
        u.best_stress = s.get("best_stress", float("inf"))
        u.best_test = s.get("best_test", float("inf"))
        u.history = s.get("history", [])
        return u


# ---------------------------
# State persistence
# ---------------------------

@dataclass
class GlobalState:
    version: str
    created_ms: int
    updated_ms: int
    base_seed: int
    task: Dict
    universes: List[Dict]
    selected_uid: int = 0
    generations_done: int = 0
    mode: str = "solver"
    rule_dsl: Optional[Dict[str, Any]] = None

STATE_DIR = Path(".rsi_state")
UPDATE_RULE_FILE = STATE_DIR / "update_rule.json"

def save_update_rule(path: Path, rule: UpdateRuleGenome):
    write_json(path, asdict(rule))

def load_update_rule(path: Path) -> UpdateRuleGenome:
    if path.exists():
        try:
            return UpdateRuleGenome.from_dict(read_json(path))
        except Exception:
            return UpdateRuleGenome.default()
    return UpdateRuleGenome.default()

def save_operators_lib(path: Path):
    path.write_text(json.dumps(OPERATORS_LIB, indent=2), encoding="utf-8")

def load_operators_lib(path: Path):
    global OPERATORS_LIB
    if path.exists():
        try:
            OPERATORS_LIB.update(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            pass

def save_state(gs: GlobalState):
    gs.updated_ms = now_ms()
    write_json(STATE_DIR / "state.json", asdict(gs))
    save_operators_lib(STATE_DIR / "operators_lib.json")
    if gs.universes:
        meta_snapshot = gs.universes[0].get("meta", {})
        if isinstance(meta_snapshot, dict) and "update_rule" in meta_snapshot:
            try:
                save_update_rule(STATE_DIR / "update_rule.json", UpdateRuleGenome.from_dict(meta_snapshot["update_rule"]))
            except Exception:
                pass
    if gs.mode == "learner":
        save_map_elites(STATE_DIR / map_elites_filename("learner"), MAP_ELITES_LEARNER)
    else:
        save_map_elites(STATE_DIR / map_elites_filename("solver"), MAP_ELITES)

def load_state() -> Optional[GlobalState]:
    p = STATE_DIR / "state.json"
    if not p.exists():
        return None
    try:
        data = read_json(p)
        mode = data.get("mode", "solver")
        load_operators_lib(STATE_DIR / "operators_lib.json")
        if mode == "learner":
            load_map_elites(STATE_DIR / map_elites_filename("learner"), MAP_ELITES_LEARNER)
        else:
            load_map_elites(STATE_DIR / map_elites_filename("solver"), MAP_ELITES)
        data["mode"] = mode
        return GlobalState(**data)
    except Exception:
        return None


def run_multiverse(
    seed: int,
    task: TaskSpec,
    gens: int,
    pop: int,
    n_univ: int,
    resume: bool = False,
    save_every: int = 5,
    mode: str = "solver",
    freeze_eval: bool = True,
) -> GlobalState:
    safe_mkdir(STATE_DIR)
    logger = RunLogger(STATE_DIR / "run_log.jsonl", append=resume)
    task.ensure_descriptor()
    update_rule = load_update_rule(STATE_DIR / "update_rule.json")

    if resume and (gs0 := load_state()):
        mode = gs0.mode
        if mode == "learner":
            us = [UniverseLearner.from_snapshot(s) for s in gs0.universes]
        else:
            us = [Universe.from_snapshot(s) for s in gs0.universes]
        for u in us:
            u.meta.update_rule = update_rule
        start = gs0.generations_done
    else:
        b0 = get_task_batch(task, seed, freeze_eval=freeze_eval)
        hint = TaskDetective.detect_pattern(b0)
        if hint:
            print(f"[Detective] Detected pattern: {hint}. Injecting smart seeds.")
        if mode == "learner":
            us = [
                UniverseLearner(
                    uid=i,
                    seed=seed + i * 9973,
                    meta=MetaState(update_rule=update_rule),
                    pool=[seed_learner_genome(random.Random(seed + i), hint) for _ in range(pop)],
                    library=FunctionLibrary(),
                )
                for i in range(n_univ)
            ]
        else:
            eval_mode = "program" if mode == "program" else ("algo" if mode == "algo" else "solver")
            us = [
                Universe(
                    uid=i,
                    seed=seed + i * 9973,
                    meta=MetaState(update_rule=update_rule),
                    pool=[seed_genome(random.Random(seed + i), hint) for _ in range(pop)],
                    library=FunctionLibrary(),
                    eval_mode=eval_mode,
                )
                for i in range(n_univ)
            ]
        start = 0

    for gen in range(start, start + gens):
        start_ms = now_ms()
        batch = get_task_batch(task, seed, freeze_eval=freeze_eval)
        for u in us:
            if mode == "learner":
                u.step(gen, task, pop, batch)
            else:
                u.step(gen, task, pop, batch)

        us.sort(key=lambda u: u.best_score)
        best = us[0]
        runtime_ms = now_ms() - start_ms
        best_code = best.best.code if best.best else "none"
        code_hash = sha256(best_code)
        novelty = 1.0 if code_hash not in logger.seen_hashes else 0.0
        logger.seen_hashes.add(code_hash)
        accepted = bool(best.history[-1]["accepted"]) if best.history else False
        last_log = best.history[-1] if best.history else {}
        control_packet = {
            "mutation_rate": best.meta.mutation_rate,
            "crossover_rate": best.meta.crossover_rate,
            "epsilon_explore": best.meta.epsilon_explore,
            "acceptance_margin": 1e-9,
            "patience": getattr(best.meta, "patience", 5),
        }
        counterexample_count = len(ALGO_COUNTEREXAMPLES.get(task.name, [])) if mode == "algo" else 0
        logger.log(
            gen=gen,
            task_id=task.name,
            mode=mode,
            score_hold=best.best_hold,
            score_stress=best.best_stress,
            score_test=getattr(best, "best_test", float("inf")),
            runtime_ms=runtime_ms,
            nodes=node_count(best_code),
            code_hash=code_hash,
            accepted=accepted,
            novelty=novelty,
            meta_policy_params={},
            solver_hash=code_hash,
            p1_hash="default",
            err_hold=best.best_hold,
            err_stress=best.best_stress,
            err_test=getattr(best, "best_test", float("inf")),
            steps=last_log.get("avg_nodes"),
            timeout_rate=last_log.get("timeout_rate"),
            counterexample_count=counterexample_count,
            library_size=len(OPERATORS_LIB),
            control_packet=control_packet,
            task_descriptor=task.descriptor.snapshot() if task.descriptor else None,
        )
        print(
            f"[Gen {gen + 1:4d}] Score: {best.best_score:.4f} | Hold: {best.best_hold:.4f} | Stress: {best.best_stress:.4f} | Test: {best.best_test:.4f} | "
            f"{(best.best.code if best.best else 'none')}"
        )

        if save_every > 0 and (gen + 1) % save_every == 0:
            gs = GlobalState(
                "RSI_EXTENDED_v2",
                now_ms(),
                now_ms(),
                seed,
                asdict(task),
                [u.snapshot() for u in us],
                us[0].uid,
                gen + 1,
                mode=mode,
            )
            save_state(gs)

    gs = GlobalState(
        "RSI_EXTENDED_v2",
        now_ms(),
        now_ms(),
        seed,
        asdict(task),
        [u.snapshot() for u in us],
        us[0].uid,
        start + gens,
        mode=mode,
    )
    save_state(gs)
    return gs


def policy_stats_from_history(history: List[Dict[str, Any]], window: int = 5) -> Dict[str, float]:
    if not history:
        return {"delta_best": 0.0, "auc_window": 0.0, "timeout_rate": 0.0, "avg_nodes": 0.0}
    holds = [h.get("hold", 0.0) for h in history]
    recent = holds[-window:] if len(holds) >= window else holds
    auc_window = sum(recent) / max(1, len(recent))
    if len(holds) >= window:
        delta_best = holds[-1] - holds[-window]
    else:
        delta_best = holds[-1] - holds[0]
    timeout_rate = history[-1].get("timeout_rate", 0.0)
    avg_nodes = history[-1].get("avg_nodes", 0.0)
    return {
        "delta_best": delta_best,
        "auc_window": auc_window,
        "timeout_rate": timeout_rate,
        "avg_nodes": avg_nodes,
    }


def run_policy_episode(
    seed: int,
    task: TaskSpec,
    policy: MetaPolicy,
    gens: int,
    pop: int,
    n_univ: int,
    freeze_eval: bool,
    library_archive: LibraryArchive,
    logger: Optional[RunLogger],
    mode: str,
    update_archive: bool = True,
) -> Tuple[List[Dict[str, Any]], Universe]:
    batch = get_task_batch(task, seed, freeze_eval=freeze_eval)
    hint = TaskDetective.detect_pattern(batch)
    descriptor = task.ensure_descriptor()
    base_lib = FunctionLibrary()
    for lib in library_archive.select(descriptor):
        base_lib.merge(lib)
    update_rule = load_update_rule(STATE_DIR / "update_rule.json")
    universes = [
        Universe(
            uid=i,
            seed=seed + i * 9973,
            meta=MetaState(update_rule=update_rule),
            pool=[seed_genome(random.Random(seed + i), hint) for _ in range(pop)],
            library=FunctionLibrary.from_snapshot(base_lib.snapshot()),
        )
        for i in range(n_univ)
    ]
    for gen in range(gens):
        start_ms = now_ms()
        stats = policy_stats_from_history(universes[0].history)
        controls = policy.act(descriptor, stats)
        for u in universes:
            u.step(gen, task, pop, batch, policy_controls=controls)
        universes.sort(key=lambda u: u.best_score)
        best = universes[0]
        if logger:
            best_code = best.best.code if best.best else "none"
            code_hash = sha256(best_code)
            novelty = 1.0 if code_hash not in logger.seen_hashes else 0.0
            logger.seen_hashes.add(code_hash)
            logger.log(
                gen=gen,
                task_id=task.name,
                mode=mode,
                score_hold=best.best_hold,
                score_stress=best.best_stress,
                score_test=best.best_test,
                runtime_ms=now_ms() - start_ms,
                nodes=node_count(best_code),
                code_hash=code_hash,
                accepted=bool(best.history[-1]["accepted"]) if best.history else False,
                novelty=novelty,
                meta_policy_params={"pid": policy.pid, "weights": policy.weights, "bias": policy.bias, "controls": controls},
                task_descriptor=descriptor.snapshot(),
            )
    universes.sort(key=lambda u: u.best_score)
    best = universes[0]
    if update_archive:
        library_archive.add(descriptor, best.best_hold, best.library)
    return best.history, best


def compute_transfer_metrics(history: List[Dict[str, Any]], window: int) -> Dict[str, float]:
    if not history:
        return {"auc": float("inf"), "regret": float("inf"), "gap": float("inf"), "recovery_time": float("inf")}
    holds = [h.get("hold", float("inf")) for h in history[:window]]
    tests = [h.get("test", float("inf")) for h in history[:window]]
    auc = sum(holds) / max(1, len(holds))
    best = min(holds)
    regret = sum(h - best for h in holds) / max(1, len(holds))
    gap = (tests[-1] - holds[-1]) if holds and tests else float("inf")
    threshold = best * 1.1 if math.isfinite(best) else float("inf")
    recovery_time = float("inf")
    for i, h in enumerate(holds):
        if h <= threshold:
            recovery_time = i + 1
            break
    return {"auc": auc, "regret": regret, "gap": gap, "recovery_time": recovery_time}


def run_meta_meta(
    seed: int,
    episodes: int,
    gens_per_episode: int,
    pop: int,
    n_univ: int,
    policy_pop: int,
    freeze_eval: bool,
    state_dir: Path,
    eval_every: int,
    few_shot_gens: int,
) -> None:
    rng = random.Random(seed)
    meta_train, meta_test = split_meta_tasks(seed)
    n_inputs = len(TaskSpec().ensure_descriptor().vector()) + 4
    policies = [MetaPolicy.seed(rng, n_outputs=5, n_inputs=n_inputs) for _ in range(policy_pop)]
    policy_scores = {p.pid: float("inf") for p in policies}
    archive = LibraryArchive(k=2)
    logger = RunLogger(state_dir / "run_log.jsonl")

    for episode in range(episodes):
        task = rng.choice(meta_train)
        policy = policies[episode % len(policies)]
        history, best = run_policy_episode(
            seed + episode * 31,
            task,
            policy,
            gens_per_episode,
            pop,
            n_univ,
            freeze_eval,
            archive,
            logger,
            mode="meta-train",
            update_archive=True,
        )
        metrics = compute_transfer_metrics(history, window=min(few_shot_gens, len(history)))
        reward = metrics["auc"]
        policy_scores[policy.pid] = min(policy_scores[policy.pid], reward)

        if (episode + 1) % eval_every == 0:
            transfer_scores = []
            for task_test in meta_test:
                warmup_task = rng.choice(meta_train) if meta_train else task_test
                warmup_gens = max(1, few_shot_gens // 2)
                run_policy_episode(
                    seed + episode * 73,
                    warmup_task,
                    policy,
                    warmup_gens,
                    pop,
                    n_univ,
                    freeze_eval,
                    archive,
                    logger,
                    mode="meta-transfer-train",
                    update_archive=True,
                )
                hist, _ = run_policy_episode(
                    seed + episode * 73 + 1,
                    task_test,
                    policy,
                    few_shot_gens,
                    pop,
                    n_univ,
                    freeze_eval,
                    archive,
                    logger,
                    mode="meta-transfer-test",
                    update_archive=False,
                )
                transfer_scores.append(compute_transfer_metrics(hist, window=few_shot_gens)["auc"])
            if transfer_scores:
                policy_scores[policy.pid] = sum(transfer_scores) / len(transfer_scores)
            policies.sort(key=lambda p: policy_scores.get(p.pid, float("inf")))
            best_policy = policies[0]
            policies = [best_policy] + [best_policy.mutate(rng, scale=0.05) for _ in range(policy_pop - 1)]


def evaluate_update_rule(
    seed: int,
    task: TaskSpec,
    rule: UpdateRuleGenome,
    gens: int,
    pop: int,
    freeze_eval: bool,
) -> Tuple[float, List[Dict[str, Any]]]:
    rng = random.Random(seed)
    batch = get_task_batch(task, seed, freeze_eval=freeze_eval)
    hint = TaskDetective.detect_pattern(batch)
    universe = Universe(
        uid=0,
        seed=seed,
        meta=MetaState(update_rule=rule),
        pool=[seed_genome(rng, hint) for _ in range(pop)],
        library=FunctionLibrary(),
    )
    for gen in range(gens):
        universe.step(gen, task, pop, batch)
    return universe.best_score, universe.history


def run_update_rule_search(
    seed: int,
    rounds: int,
    gens_per_round: int,
    pop: int,
    freeze_eval: bool,
    state_dir: Path,
) -> UpdateRuleGenome:
    rng = random.Random(seed)
    task = TaskSpec(name="self_audit", n_train=64, n_hold=48, n_test=48, noise=0.0)
    current = load_update_rule(state_dir / "update_rule.json")
    rules = [current] + [current.mutate(rng) for _ in range(3)]
    best_rule = current
    best_score = float("inf")
    for r in range(rounds):
        scored: List[Tuple[float, UpdateRuleGenome]] = []
        for idx, rule in enumerate(rules):
            score, history = evaluate_update_rule(seed + r * 101 + idx, task, rule, gens_per_round, pop, freeze_eval)
            scored.append((score, rule))
            if history:
                last = history[-1]
                if last.get("code"):
                    SURROGATE.train(history)
        scored.sort(key=lambda item: item[0])
        score, best = scored[0]
        if score < best_score:
            best_score = score
            best_rule = best
            save_update_rule(state_dir / "update_rule.json", best_rule)
        rules = [best] + [best.mutate(rng) for _ in range(max(1, len(rules) - 1))]
    return best_rule


def run_task_switch(
    seed: int,
    task_a: TaskSpec,
    task_b: TaskSpec,
    gens_a: int,
    gens_b: int,
    pop: int,
    n_univ: int,
    freeze_eval: bool,
    state_dir: Path,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    n_inputs = len(TaskSpec().ensure_descriptor().vector()) + 4
    transfer_policy = MetaPolicy.seed(rng, n_outputs=5, n_inputs=n_inputs)
    archive = LibraryArchive(k=2)
    logger = RunLogger(state_dir / "run_log.jsonl")
    baseline = MetaPolicy.seed(random.Random(seed + 999), n_outputs=5, n_inputs=n_inputs)

    history_a, _ = run_policy_episode(
        seed,
        task_a,
        transfer_policy,
        gens_a,
        pop,
        n_univ,
        freeze_eval,
        archive,
        logger,
        mode="switch-train",
        update_archive=True,
    )
    history_transfer, _ = run_policy_episode(
        seed + 1,
        task_b,
        transfer_policy,
        gens_b,
        pop,
        n_univ,
        freeze_eval,
        archive,
        logger,
        mode="switch-transfer",
        update_archive=False,
    )
    history_baseline, _ = run_policy_episode(
        seed + 2,
        task_b,
        baseline,
        gens_b,
        pop,
        n_univ,
        freeze_eval,
        LibraryArchive(k=0),
        logger,
        mode="switch-baseline",
        update_archive=False,
    )
    metrics_transfer = compute_transfer_metrics(history_transfer, window=gens_b)
    metrics_baseline = compute_transfer_metrics(history_baseline, window=gens_b)
    delta_auc = metrics_baseline["auc"] - metrics_transfer["auc"]
    delta_recovery = metrics_baseline["recovery_time"] - metrics_transfer["recovery_time"]
    return {
        "transfer": metrics_transfer,
        "baseline": metrics_baseline,
        "delta_auc": delta_auc,
        "delta_recovery_time": delta_recovery,
    }


def generate_report(path: Path, few_shot_gens: int) -> Dict[str, Any]:
    if not path.exists():
        return {"error": "run_log.jsonl not found"}
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    by_task: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        key = f"{rec['task_id']}::{rec.get('mode', 'unknown')}"
        by_task.setdefault(key, []).append(rec)
    report = {"tasks": {}, "few_shot_gens": few_shot_gens}
    for key, recs in by_task.items():
        recs.sort(key=lambda r: r["gen"])
        holds = [r["score_hold"] for r in recs[:few_shot_gens]]
        tests = [r["score_test"] for r in recs[:few_shot_gens]]
        auc = sum(holds) / max(1, len(holds))
        best = min(holds) if holds else float("inf")
        regret = sum(h - best for h in holds) / max(1, len(holds))
        gap = (tests[-1] - holds[-1]) if holds and tests else float("inf")
        threshold = best * 1.1 if math.isfinite(best) else float("inf")
        recovery_time = float("inf")
        for i, h in enumerate(holds):
            if h <= threshold:
                recovery_time = i + 1
                break
        few_shot_delta = (holds[0] - holds[-1]) if len(holds) > 1 else 0.0
        report["tasks"][key] = {
            "auc": auc,
            "regret": regret,
            "generalization_gap": gap,
            "recovery_time": recovery_time,
            "few_shot_delta": few_shot_delta,
        }
    return report


def transfer_bench(
    task_from: str,
    task_to: str,
    budget: int,
    seed: int,
    freeze_eval: bool = True,
) -> Dict[str, Any]:
    task_a = TaskSpec(name=task_from)
    task_b = TaskSpec(name=task_to)
    mode = "algo" if task_from in ALGO_TASK_NAMES else "solver"
    u = Universe(uid=0, seed=seed, meta=MetaState(), pool=[], library=FunctionLibrary(), eval_mode=mode)
    rng = random.Random(seed)

    for g in range(budget):
        batch = get_task_batch(task_a, seed, freeze_eval=freeze_eval, gen=g)
        if batch is None:
            break
        u.step(g, task_a, 24, batch)

    holds: List[float] = []
    for g in range(budget):
        batch = get_task_batch(task_b, seed + 17, freeze_eval=freeze_eval, gen=g)
        if batch is None:
            break
        u.step(g, task_b, 24, batch)
        holds.append(u.best_hold)

    auc = sum(holds) / max(1, len(holds))
    best = min(holds) if holds else float("inf")
    threshold = best * 1.1 if math.isfinite(best) else float("inf")
    recovery_time = float("inf")
    for i, h in enumerate(holds):
        if h <= threshold:
            recovery_time = i + 1
            break

    record = {
        "from": task_from,
        "to": task_to,
        "budget": budget,
        "seed": seed,
        "auc_N": auc,
        "recovery_time": recovery_time,
        "holds": holds,
    }
    out = STATE_DIR / "transfer_bench.jsonl"
    safe_mkdir(out.parent)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return record


# ---------------------------
# RSI Loop (Hard Gates + Rollback)
# ---------------------------

STRESS_MAX = 1_000_000.0
OUTPUT_VARIANCE_EPS = 1e-6
RSI_CONFIRM_ROUNDS = 2

def _outputs_constant(outputs: List[Any], tol: float = 1e-9) -> bool:
    if not outputs:
        return True
    first = outputs[0]
    if isinstance(first, (int, float)):
        return all(isinstance(o, (int, float)) and abs(o - first) <= tol for o in outputs[1:])
    return all(_algo_equal(o, first) for o in outputs[1:])

def _unique_output_count(outputs: List[Any]) -> int:
    uniques: List[Any] = []
    for out in outputs:
        if not any(_algo_equal(out, seen) for seen in uniques):
            uniques.append(out)
    return len(uniques)

def _piecewise_constant(outputs: List[Any], max_unique: int = 2) -> bool:
    if not outputs:
        return True
    return _unique_output_count(outputs) <= max_unique

def _variance_low(outputs: List[Any], eps: float = OUTPUT_VARIANCE_EPS) -> bool:
    if not outputs or not all(isinstance(o, (int, float)) for o in outputs):
        return False
    vals = [float(o) for o in outputs]
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / max(1, len(vals))
    return var <= eps

def _collect_outputs(
    code: str,
    xs: List[Any],
    mode: str,
    extra_env: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, List[Any], str]:
    outputs: List[Any] = []
    if mode == "learner":
        env = safe_load_module(code)
        if not env:
            return False, [], "load_failed"
        required = ["init_mem", "encode", "predict"]
        if not all(name in env and callable(env[name]) for name in required):
            return False, [], "missing_funcs"
        mem = env["init_mem"]()
        encode = env["encode"]
        predict = env["predict"]
        for x in xs:
            try:
                z = encode(x, mem)
                out = predict(z, mem)
            except Exception:
                return False, [], "exec_error"
            outputs.append(out)
        return True, outputs, ""
    if mode == "algo":
        for x in xs:
            out, _, timeout = safe_exec_algo(code, x)
            if timeout:
                return False, [], "timeout"
            outputs.append(out)
        return True, outputs, ""
    for x in xs:
        out = safe_exec(code, x, extra_env=extra_env)
        if out is None:
            return False, [], "no_output"
        outputs.append(out)
    return True, outputs, ""

def _hard_gate_ok(
    code: str,
    batch: Batch,
    mode: str,
    task_name: str,
    extra_env: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str]:
    xs = batch.x_ho[:8] if batch.x_ho else batch.x_tr[:8]
    if not xs:
        return False, "no_inputs"
    ok, outputs, err = _collect_outputs(code, xs, mode, extra_env=extra_env)
    if not ok:
        return False, err
    # Hard gate: reject any non-finite numeric output (timeouts/NaNs are disqualifying).
    for out in outputs:
        if isinstance(out, (int, float)) and not math.isfinite(out):
            return False, "non_finite_output"
    # Hard gate: reject constant or near-constant outputs to enforce input dependence.
    if _outputs_constant(outputs):
        return False, "constant_output"
    # Hard gate: prevent piecewise-constant or low-diversity output hacks.
    if _piecewise_constant(outputs):
        return False, "piecewise_constant"
    # Hard gate: reject numerically low-variance responses (e.g., tiny jitter around a constant).
    if _variance_low(outputs):
        return False, "low_variance_output"
    return True, ""

def _evaluate_candidate(
    g: Union[Genome, LearnerGenome],
    batch: Batch,
    mode: str,
    task_name: str,
    extra_env: Optional[Dict[str, Any]] = None,
    validator: Callable[[str], Tuple[bool, str]] = validate_code,
) -> EvalResult:
    gate_ok, gate_reason = _hard_gate_ok(g.code, batch, mode, task_name, extra_env=extra_env)
    if not gate_ok:
        return EvalResult(
            False,
            float("inf"),
            float("inf"),
            float("inf"),
            float("inf"),
            node_count(g.code),
            float("inf"),
            f"hard_gate:{gate_reason}",
        )
    if mode == "learner":
        return evaluate_learner(g, batch, task_name)
    if mode == "algo":
        return evaluate_algo(g, batch, task_name)
    return evaluate(g, batch, task_name, extra_env=extra_env, validator=validator)

def _merge_stress(fixed: Batch, resampled: Batch) -> Batch:
    return Batch(
        x_tr=resampled.x_tr,
        y_tr=resampled.y_tr,
        x_ho=resampled.x_ho,
        y_ho=resampled.y_ho,
        x_st=fixed.x_st + resampled.x_st,
        y_st=fixed.y_st + resampled.y_st,
        x_te=resampled.x_te,
        y_te=resampled.y_te,
    )

def _load_rsi_archive(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"entries": [], "current": None, "consecutive": 0}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"entries": [], "current": None, "consecutive": 0}

def _save_rsi_archive(path: Path, archive: Dict[str, Any]) -> None:
    safe_mkdir(path.parent)
    path.write_text(json.dumps(archive, indent=2), encoding="utf-8")

def _load_state_snapshot(state_dir: Path) -> Optional[Dict[str, Any]]:
    state_path = state_dir / "state.json"
    if not state_path.exists():
        return None
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return None

def _write_state_snapshot(state_dir: Path, snapshot: Dict[str, Any]) -> None:
    safe_mkdir(state_dir)
    (state_dir / "state.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

def _current_best_score(snapshot: Optional[Dict[str, Any]]) -> float:
    if not snapshot:
        return float("inf")
    universes = snapshot.get("universes", [])
    selected_uid = snapshot.get("selected_uid", None)
    best = float("inf")
    for u in universes:
        score = float(u.get("best_score", float("inf")))
        if selected_uid is not None and u.get("uid") == selected_uid:
            return score
        best = min(best, score)
    return best

def _clone_state_dir(src: Path, dest: Path) -> None:
    safe_mkdir(dest)
    if not src.exists():
        return
    for item in src.iterdir():
        target = dest / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)

def _autopatch_evolve_score(
    script: Path,
    state_dir: Path,
    mode: str,
    task_name: str,
    seed: int,
    generations: int,
    population: int,
    universes: int,
    resume: bool,
    freeze_eval: bool = True,
) -> float:
    if mode == "learner":
        cmd = [
            sys.executable,
            str(script),
            "learner-evolve",
            "--seed",
            str(seed),
            "--generations",
            str(generations),
            "--population",
            str(population),
            "--universes",
            str(universes),
            "--task",
            task_name,
            "--state-dir",
            str(state_dir),
        ]
    else:
        cmd = [
            sys.executable,
            str(script),
            "evolve",
            "--seed",
            str(seed),
            "--generations",
            str(generations),
            "--population",
            str(population),
            "--universes",
            str(universes),
            "--task",
            task_name,
            "--state-dir",
            str(state_dir),
        ]
        if mode:
            cmd.extend(["--mode", mode])
    if resume:
        cmd.append("--resume")
    if not freeze_eval:
        cmd.append("--no-freeze-eval")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        return float("inf")
    snapshot = _load_state_snapshot(state_dir)
    if not snapshot:
        return float("inf")
    return _current_best_score(snapshot)

def _autopatch_probe_score(
    mode: str,
    task_name: str,
    seed: int = 1337,
    generations: int = 6,
    population: int = 32,
    universes: int = 1,
    freeze_eval: bool = True,
) -> float:
    task = TaskSpec(name=task_name)
    gs = run_multiverse(
        seed,
        task,
        generations,
        population,
        universes,
        resume=False,
        save_every=0,
        mode=mode,
        freeze_eval=freeze_eval,
    )
    if not gs.universes:
        return float("inf")
    best_snapshot = next((u for u in gs.universes if u.get("uid") == gs.selected_uid), gs.universes[0])
    return float(best_snapshot.get("best_score", float("inf")))

def _probe_score(script: Path, mode: str, task_name: str) -> float:
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "autopatch-probe",
                "--mode",
                mode,
                "--task",
                task_name,
                "--state-dir",
                tmpdir,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
    if result.returncode != 0:
        return float("inf")
    output = result.stdout.strip().splitlines()
    if not output:
        return float("inf")
    try:
        return float(output[-1].strip())
    except Exception:
        return float("inf")

def _replace_source_segment(source: str, old: str, new: str) -> str:
    if old not in source:
        return source
    return source.replace(old, new, 1)

def _mutate_hyperparameter(
    tree: ast.AST,
    source: str,
    param_name: str,
    rng: random.Random,
) -> Tuple[str, str, Optional[float]]:
    ranges: Dict[str, Tuple[float, float]] = {
        "mutation_rate": (0.05, 0.95),
        "crossover_rate": (0.0, 0.9),
        "complexity_lambda": (1e-5, 1e-2),
        "epsilon_explore": (0.05, 0.5),
    }
    int_ranges: Dict[str, Tuple[int, int]] = {
        "adapt_steps": (4, 16),
    }
    target_node: Optional[ast.AnnAssign] = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "MetaState":
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    if stmt.target.id == param_name:
                        target_node = stmt
                        break
    if not target_node:
        return source, "", None
    old_segment = ast.get_source_segment(source, target_node.value) or ""
    if param_name in int_ranges:
        low, high = int_ranges[param_name]
        new_value = rng.randint(low, high)
        new_segment = str(new_value)
    else:
        low, high = ranges.get(param_name, (0.0, 1.0))
        new_value = rng.uniform(low, high)
        new_segment = f"{new_value:.6f}"
    new_source = _replace_source_segment(source, old_segment, new_segment)
    return new_source, f"L1:{param_name}", float(new_value)

def _mutate_operator(tree: ast.AST, source: str, rng: random.Random) -> Tuple[str, str]:
    target_assign: Optional[ast.Assign] = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "act":
            for stmt in node.body:
                if isinstance(stmt, ast.Assign):
                    if len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                        if stmt.targets[0].id == "op_weights" and isinstance(stmt.value, ast.Dict):
                            target_assign = stmt
                            break
    if not target_assign or not isinstance(target_assign.value, ast.Dict):
        return source, ""
    new_source = source
    for key_node, value_node in zip(target_assign.value.keys, target_assign.value.values):
        if not isinstance(key_node, ast.Constant) or not isinstance(key_node.value, str):
            continue
        key = key_node.value
        if key not in ("insert_assign", "list_manip"):
            continue
        old_segment = ast.get_source_segment(source, value_node) or ""
        offset = rng.uniform(0.0, 0.5)
        if key == "list_manip":
            offset = rng.uniform(0.0, 0.3)
        if "op_scale" not in old_segment:
            continue
        if "op_scale +" in old_segment:
            new_segment = re.sub(r"op_scale\s*\+\s*[-+]?\d*\.?\d+", f"op_scale + {offset:.3f}", old_segment)
        else:
            new_segment = re.sub(r"op_scale\s*-\s*[-+]?\d*\.?\d+", f"op_scale - {offset:.3f}", old_segment)
        if new_segment == old_segment:
            continue
        new_source = _replace_source_segment(new_source, old_segment, new_segment)
    return new_source, "L2:op_weights"

def _mutate_evaluation(tree: ast.AST, source: str, rng: random.Random) -> Tuple[str, str]:
    weights = {
        "SCORE_W_HOLD": rng.uniform(0.45, 0.7),
        "SCORE_W_STRESS": rng.uniform(0.2, 0.6),
        "SCORE_W_TRAIN": rng.uniform(0.0, 0.2),
    }
    new_source = source
    changed = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in weights:
                old_segment = ast.get_source_segment(source, node.value) or ""
                new_segment = f"{weights[name]:.6f}"
                if old_segment:
                    new_source = _replace_source_segment(new_source, old_segment, new_segment)
                    changed = True
    return new_source, "L3:score_weights" if changed else ""

def _evaluate_patch_candidate(
    patch_code: str,
    baseline_score: float,
    mode: str,
    task_name: str,
) -> Tuple[bool, float, float]:
    """Evaluate a patch candidate. Returns (accepted, improvement, new_score)."""
    min_improvement_threshold = 0.03
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(patch_code)
        tmp = Path(f.name)
    try:
        new_score = _probe_score(tmp, mode, task_name)
    finally:
        tmp.unlink(missing_ok=True)
    if not math.isfinite(baseline_score) or baseline_score <= 0:
        return False, 0.0, new_score
    improvement = (baseline_score - new_score) / baseline_score
    return improvement >= min_improvement_threshold, improvement, new_score

def _select_best_patch(candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    valid = [c for c in candidates if c.get("improvement", 0.0) > 0.0]
    if not valid:
        return None
    return max(valid, key=lambda c: (c["improvement"], -c["diff_size"]))

def _safe_apply_patch(self_path: Path, new_code: str) -> bool:
    backup_path = self_path.with_suffix(".py.bak")
    shutil.copy(self_path, backup_path)
    try:
        ast.parse(new_code)
        self_path.write_text(new_code, encoding="utf-8")
        result = subprocess.run(
            [sys.executable, str(self_path), "selftest"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError("Selftest failed")
        return True
    except Exception:
        shutil.copy(backup_path, self_path)
        return False

def _log_autopatch_attempt(record: Dict[str, Any]) -> None:
    log_path = STATE_DIR / "autopatch_log.jsonl"
    safe_mkdir(log_path.parent)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

def run_deep_autopatch(
    levels: List[int],
    candidates: int = 4,
    apply: bool = True,
    mode: str = "solver",
) -> Dict[str, Any]:
    """
    True RSI self-modification system with fitness-gated acceptance and rollback safety.
    Core change: evolve after mutation instead of re-evaluating the same code.
    """
    script = Path(__file__).resolve()
    source = script.read_text(encoding="utf-8")
    state_snapshot = _load_state_snapshot(STATE_DIR)
    task_name = (state_snapshot or {}).get("task", {}).get("name", TaskSpec().name)
    seed = int((state_snapshot or {}).get("base_seed", 1337))
    universes = max(1, len((state_snapshot or {}).get("universes", [])) or 1)
    pool_len = 0
    if state_snapshot and state_snapshot.get("universes"):
        pool_len = len(state_snapshot["universes"][0].get("pool", []))
    population = max(64, pool_len, 32)
    baseline = _current_best_score(state_snapshot)
    if not math.isfinite(baseline) or baseline <= 0:
        baseline = _probe_score(script, mode, task_name)
    print(f"[AUTOPATCH L{levels}] Baseline: {baseline:.4f}")

    rng = random.Random(int(time.time()) % 100000)
    patch_candidates: List[Dict[str, Any]] = []
    attempt_idx = 0

    for level in levels:
        for _ in range(candidates):
            attempt_idx += 1
            tree = ast.parse(source)
            patch_type = ""
            mutated_source = source
            mutated_state = copy.deepcopy(state_snapshot) if state_snapshot else None
            mutated_params: Dict[str, Any] = {}
            if level == 1:
                param = rng.choice(["mutation_rate", "crossover_rate", "complexity_lambda", "epsilon_explore", "adapt_steps"])
                mutated_source, patch_type, new_value = _mutate_hyperparameter(tree, source, param, rng)
                if new_value is None:
                    continue
                if param == "adapt_steps":
                    new_value = int(round(new_value))
                mutated_params[param] = new_value
                if mutated_state:
                    for u in mutated_state.get("universes", []):
                        meta = u.get("meta", {})
                        meta[param] = new_value
                        u["meta"] = meta
            elif level == 2:
                mutated_source, patch_type = _mutate_operator(tree, source, rng)
            elif level == 3:
                mutated_source, patch_type = _mutate_evaluation(tree, source, rng)
            if not patch_type or (mutated_source == source and not mutated_params):
                continue
            diff = unified_diff(source, mutated_source, str(script))
            diff_size = len(diff.splitlines())
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_state_dir = Path(tmpdir)
                if state_snapshot:
                    _clone_state_dir(STATE_DIR, tmp_state_dir)
                    if mutated_state:
                        _write_state_snapshot(tmp_state_dir, mutated_state)
                script_path = script
                if mutated_source != source:
                    script_path = tmp_state_dir / script.name
                    script_path.write_text(mutated_source, encoding="utf-8")
                attempt_seed = seed + attempt_idx
                print(f"[DEBUG] Running evolution with params: {mutated_params}")
                new_score = _autopatch_evolve_score(
                    script_path,
                    tmp_state_dir,
                    mode,
                    task_name,
                    attempt_seed,
                    generations=15,
                    population=population,
                    universes=universes,
                    resume=state_snapshot is not None,
                    freeze_eval=True,
                )
                print(f"[DEBUG] Evolution returned best_score: {new_score}")
            improvement = baseline - new_score
            accepted = improvement > 0
            record = {
                "level": level,
                "patch_type": patch_type,
                "old_score": baseline,
                "new_score": new_score,
                "improvement": improvement,
                "diff_size": diff_size,
                "accepted": accepted,
                "params": mutated_params,
            }
            _log_autopatch_attempt(record)
            if accepted:
                print(f"[AUTOPATCH] {patch_type} -> {new_score:.4f} (ACCEPT +{improvement:.2f})")
            else:
                print(f"[AUTOPATCH] {patch_type} -> {new_score:.4f} (REJECT)")
            patch_candidates.append(
                {
                    **record,
                    "diff": diff,
                    "code": mutated_source,
                    "state": mutated_state,
                }
            )

    best = _select_best_patch(patch_candidates)
    if not best:
        return {
            "applied": False,
            "improvement": 0.0,
            "old_score": baseline,
            "new_score": baseline,
            "patch_type": "",
            "diff": "",
    }

    if apply:
        applied = True
        if best["code"] != source:
            applied = _safe_apply_patch(script, best["code"])
        if applied and best.get("state"):
            _write_state_snapshot(STATE_DIR, best["state"])
        if applied:
            print(f"[RSI] Self-modified! Score: {best['old_score']:.4f} -> {best['new_score']:.4f}")
        return {
            "applied": applied,
            "improvement": best["improvement"],
            "old_score": best["old_score"],
            "new_score": best["new_score"],
            "patch_type": best["patch_type"],
            "diff": best["diff"],
        }

    return {
        "applied": False,
        "improvement": best["improvement"],
        "old_score": best["old_score"],
        "new_score": best["new_score"],
        "patch_type": best["patch_type"],
        "diff": best["diff"],
    }


def load_recent_scores(log_path: Path, n: int) -> List[float]:
    scores = []
    if not log_path.exists():
        return scores
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines[-n:]:
                try:
                    data = json.loads(line)
                    if "score" in data:
                        scores.append(float(data["score"]))
                except:
                    pass
    except Exception:
        pass
    return scores


def is_300s_stagnation(scores: List[float]) -> bool:
    if len(scores) < 5:
        return False
    return all(s > 300.0 for s in scores)


def _candidate_hash(code: str) -> str:
    return sha256(code)


def _slice_pair(xs: List[Any], ys: List[Any], n: int) -> Tuple[List[Any], List[Any]]:
    if not xs or not ys:
        return [], []
    k = min(n, len(xs), len(ys))
    return xs[:k], ys[:k]


def _prefilter_batch(batch: Batch, max_samples: int = 3) -> Batch:
    x_tr, y_tr = _slice_pair(batch.x_tr, batch.y_tr, max_samples)
    x_ho, y_ho = _slice_pair(batch.x_ho, batch.y_ho, max_samples)
    x_st, y_st = _slice_pair(batch.x_st, batch.y_st, max_samples)
    x_te, y_te = _slice_pair(batch.x_te, batch.y_te, max_samples)
    if not x_tr and x_ho:
        x_tr, y_tr = x_ho, y_ho
    if not x_ho and x_tr:
        x_ho, y_ho = x_tr, y_tr
    if not x_st and x_tr:
        x_st, y_st = x_tr, y_tr
    if not x_te and x_tr:
        x_te, y_te = x_tr, y_tr
    return Batch(x_tr, y_tr, x_ho, y_ho, x_st, y_st, x_te, y_te)


def _prefilter_eval(
    g: Genome,
    batch: Batch,
    mode: str,
    task_name: str,
    extra_env: Optional[Dict[str, Any]] = None,
    validator: Callable[[str], Tuple[bool, str]] = validate_code,
) -> Tuple[bool, str, Optional[EvalResult]]:
    gate_ok, gate_reason = _hard_gate_ok(g.code, batch, mode, task_name, extra_env=extra_env)
    if not gate_ok:
        return False, f"hard_gate:{gate_reason}", None
    if mode in ("solver", "program"):
        ok, err = validator(g.code)
        if not ok:
            return False, f"validator:{err}", None
    mini_batch = _prefilter_batch(batch, max_samples=4)
    if mode == "learner":
        res = evaluate_learner(g, mini_batch, task_name)
    elif mode == "algo":
        res = evaluate_algo(g, mini_batch, task_name)
    else:
        res = evaluate(g, mini_batch, task_name, extra_env=extra_env, validator=validator)
    return res.ok, res.err or "", res


def _mutate_genome_with_meta(
    rng: random.Random,
    g: Genome,
    meta: MetaState,
    library: FunctionLibrary,
    op_bias: Optional[str] = None,
) -> Genome:
    stmts = g.statements[:]
    op_tag = "mutate"
    use_synth = rng.random() < 0.3 and bool(OPERATORS_LIB)
    if use_synth:
        synth_name = rng.choice(list(OPERATORS_LIB.keys()))
        steps = OPERATORS_LIB[synth_name].get("steps", [])
        stmts = apply_synthesized_op(rng, stmts, steps)
        op_tag = f"synth:{synth_name}"
    else:
        op = op_bias or meta.sample_op(rng)
        if op in OPERATORS:
            stmts = OPERATORS[op](rng, stmts)
        op_tag = f"mut:{op}"
    stmts = inject_helpers_into_statements(rng, list(stmts), library)
    return Genome(statements=stmts, parents=[g.gid], op_tag=op_tag)


def _synthesize_genome(
    rng: random.Random,
    pool: List[Genome],
    hint: Optional[str],
    library: FunctionLibrary,
) -> Genome:
    if not pool:
        return seed_genome(rng, hint)
    p1 = rng.choice(pool)
    p2 = rng.choice(pool)
    if len(p1.statements) <= 1 or len(p2.statements) <= 1:
        stmts = (p1.statements or []) + (p2.statements or [])
    else:
        cut1 = max(1, len(p1.statements) // 2)
        cut2 = max(1, len(p2.statements) // 2)
        stmts = p1.statements[:cut1] + p2.statements[-cut2:]
    if not stmts:
        stmts = ["return x"]
    stmts = inject_helpers_into_statements(rng, list(stmts), library)
    return Genome(statements=stmts, parents=[p1.gid, p2.gid], op_tag="synthesize")


def _fallback_template_genome(rng: random.Random, hint: Optional[str]) -> Genome:
    if hint:
        return seed_genome(rng, hint)
    return Genome(statements=["v0 = x", "return v0"], op_tag="fallback")


def _simplify_genome(rng: random.Random, g: Genome) -> Optional[Genome]:
    if len(g.statements) <= 1:
        return None
    stmts = g.statements[:]
    removable = [i for i, s in enumerate(stmts) if not s.strip().startswith("return ")]
    if not removable:
        return None
    idx = rng.choice(removable)
    del stmts[idx]
    return Genome(statements=stmts, parents=[g.gid], op_tag="simplify")


def _repair_genome(g: Genome) -> Genome:
    stmts = g.statements[:]
    has_return = any(s.strip().startswith("return ") for s in stmts)
    if not has_return:
        stmts.append("return x")
    else:
        if not any("x" in s for s in stmts if s.strip().startswith("return ")):
            stmts.append("return x")
    return Genome(statements=stmts, parents=[g.gid], op_tag="repair")


def _critic_refine(
    rng: random.Random,
    g: Genome,
    meta: MetaState,
    library: FunctionLibrary,
) -> List[Genome]:
    refined: List[Genome] = []
    simplified = _simplify_genome(rng, g)
    if simplified:
        refined.append(simplified)
    refined.append(_repair_genome(g))
    refined.append(_mutate_genome_with_meta(rng, g, meta, library, op_bias="modify_return"))
    return refined


def _adjust_creator_policy(
    policy: AgentPolicy,
    gate_pass_rate: float,
    gate_fail_reasons: collections.Counter,
) -> AgentPolicy:
    new_search = dict(policy.search_bias)
    generator_mode = policy.generator_mode
    if gate_pass_rate < policy.gate_target:
        generator_mode = "template"
        new_search["simplicity"] = clamp(new_search.get("simplicity", 0.5) + 0.3, 0.1, 2.0)
    if gate_fail_reasons.get("constant_output", 0) > 0:
        new_search["robustness"] = clamp(new_search.get("robustness", 0.5) + 0.2, 0.1, 2.0)
    return AgentPolicy(
        generator_mode=generator_mode,
        search_bias=new_search,
        gate_target=policy.gate_target,
        slice_seconds=policy.slice_seconds,
    )


def _critic_rank_score(res: EvalResult, policy: AgentPolicy) -> float:
    simplicity = policy.search_bias.get("simplicity", 0.0)
    robustness = policy.search_bias.get("robustness", 0.0)
    generalization = policy.search_bias.get("generalization", 0.0)
    perf = policy.search_bias.get("perf", 0.0)
    return (
        res.score
        + simplicity * 0.0005 * res.nodes
        + robustness * res.stress
        + generalization * res.hold
        + perf * res.train
    )


def _print_critic_summary(
    gate_pass: int,
    total_checked: int,
    adopted: bool,
    full_results_count: int,
    duplicate_count: int,
    scored_empty_count: int,
    gate_fail_reasons: collections.Counter,
    validator_fail_reasons: collections.Counter,
) -> None:
    gate_pass_rate = gate_pass / max(1, total_checked)
    adoption_rate = (1.0 if adopted else 0.0) / max(1, full_results_count)
    duplicate_ratio = duplicate_count / max(1, total_checked)
    top_gate = gate_fail_reasons.most_common(5)
    print(
        f"[Critic] gate_pass_rate={gate_pass_rate:.2f} adoption_rate={adoption_rate:.2f} "
        f"duplicate_ratio={duplicate_ratio:.2f} scored_empty={scored_empty_count}"
    )
    if top_gate:
        print("[Critic] top gate failures:", ", ".join(f"{k}:{v}" for k, v in top_gate))
    else:
        print("[Critic] top gate failures: none")
    if validator_fail_reasons:
        top_validator = validator_fail_reasons.most_common(3)
        print("[Critic] top validator failures:", ", ".join(f"{k}:{v}" for k, v in top_validator))


def run_duo_loop(
    rounds: int,
    slice_seconds: float,
    blackboard_path: Path,
    k_full: int,
    seed: int,
    mode: str = "solver",
    freeze_eval: bool = True,
    population: int = 64,
    max_candidates: int = 512,
) -> None:
    task = TaskSpec()
    task.ensure_descriptor()
    rng = random.Random(seed)
    gs = load_state()
    if gs and gs.mode == mode and gs.universes:
        selected = next((u for u in gs.universes if u.get("uid") == gs.selected_uid), gs.universes[0])
        universe = Universe.from_snapshot(selected)
    else:
        batch0 = get_task_batch(task, seed, freeze_eval=freeze_eval)
        hint = TaskDetective.detect_pattern(batch0)
        universe = Universe(
            uid=0,
            seed=seed,
            meta=MetaState(),
            pool=[seed_genome(random.Random(seed + i), hint) for i in range(population)],
            library=FunctionLibrary(),
            eval_mode="program" if mode == "program" else ("algo" if mode == "algo" else "solver"),
        )

    creator_policy = CREATOR_POLICY
    critic_policy = CRITIC_POLICY
    reseed_templates: List[List[str]] = []
    fixed_batch = get_task_batch(task, seed, freeze_eval=freeze_eval, gen=0)
    if fixed_batch is None:
        print("[DUO] No batch available; aborting.")
        return

    for r in range(rounds):
        round_seed = seed + r * 9973
        round_rng = random.Random(round_seed)
        batch = get_task_batch(task, seed, freeze_eval=freeze_eval, gen=r)
        if batch is None:
            print("[DUO] No batch available; aborting.")
            break
        helper_env = universe.library.get_helpers()
        hint = TaskDetective.detect_pattern(batch)
        if hint:
            print(f"[DUO] Detected pattern: {hint}")

        creator_slice = slice_seconds if slice_seconds > 0 else creator_policy.slice_seconds
        critic_slice = slice_seconds if slice_seconds > 0 else critic_policy.slice_seconds

        print(f"\n{'='*60}\n[DUO ROUND {r+1}/{rounds}] Creator\n{'='*60}")
        creator_candidates: List[Genome] = [
            seed_genome(round_rng, hint),
            _fallback_template_genome(round_rng, hint),
        ]
        if universe.best:
            creator_candidates.append(_repair_genome(universe.best))
        creator_start = time.time()
        while time.time() - creator_start < creator_slice:
            if len(creator_candidates) >= max_candidates:
                break
            mode_choice = creator_policy.generator_mode
            if mode_choice == "template":
                if reseed_templates:
                    stmts = round_rng.choice(reseed_templates)
                    g = Genome(statements=list(stmts), op_tag="reseed")
                else:
                    g = seed_genome(round_rng, hint)
            elif mode_choice == "mutate":
                parent = round_rng.choice(universe.pool) if universe.pool else seed_genome(round_rng, hint)
                g = _mutate_genome_with_meta(round_rng, parent, universe.meta, universe.library)
            else:
                g = _synthesize_genome(round_rng, universe.pool, hint, universe.library)
            creator_candidates.append(g)

        print(f"[DUO] Creator proposed {len(creator_candidates)} candidates")

        print(f"\n{'='*60}\n[DUO ROUND {r+1}/{rounds}] Critic\n{'='*60}")
        critic_start = time.time()
        gate_fail_reasons: collections.Counter = collections.Counter()
        validator_fail_reasons: collections.Counter = collections.Counter()
        scored_empty_count = 0
        prefiltered: List[Tuple[Genome, EvalResult]] = []
        seen_hashes: Set[str] = set()
        duplicate_count = 0
        total_checked = 0
        gate_pass = 0

        for g in creator_candidates:
            if time.time() - critic_start > critic_slice:
                break
            total_checked += 1
            code_hash = _candidate_hash(g.code)
            if code_hash in seen_hashes:
                duplicate_count += 1
            seen_hashes.add(code_hash)

            ok, reason, pre_res = _prefilter_eval(
                g,
                batch,
                universe.eval_mode,
                task.name,
                extra_env=helper_env,
                validator=validate_program if universe.eval_mode == "program" else validate_code,
            )
            record = {
                "timestamp": now_ms(),
                "agent_id": "critic",
                "generation": r,
                "candidate_hash": code_hash,
                "gate_ok": ok,
                "gate_reason": "" if ok else reason,
                "score_train": pre_res.train if pre_res else None,
                "score_holdout": pre_res.hold if pre_res else None,
                "score_stress": pre_res.stress if pre_res else None,
                "selected": False,
                "note": "prefilter",
            }
            append_blackboard(blackboard_path, record)

            if not ok:
                if reason.startswith("hard_gate:"):
                    gate_fail_reasons[reason.split("hard_gate:", 1)[1]] += 1
                elif reason.startswith("validator:"):
                    validator_fail_reasons[reason.split("validator:", 1)[1]] += 1
                continue
            gate_pass += 1
            if pre_res:
                prefiltered.append((g, pre_res))

        if not prefiltered:
            scored_empty_count += 1
            reseed_templates = [_fallback_template_genome(round_rng, hint).statements]
            append_blackboard(
                blackboard_path,
                {
                    "timestamp": now_ms(),
                    "agent_id": "critic",
                    "generation": r,
                    "candidate_hash": "none",
                    "gate_ok": False,
                    "gate_reason": "scored_empty",
                    "score_train": None,
                    "score_holdout": None,
                    "score_stress": None,
                    "selected": False,
                    "note": "reseed",
                },
            )
            gate_pass_rate = gate_pass / max(1, total_checked)
            creator_policy = _adjust_creator_policy(creator_policy, gate_pass_rate, gate_fail_reasons)
            print("[DUO] No candidates passed prefilter; reseeding templates.")
            _print_critic_summary(
                gate_pass=gate_pass,
                total_checked=total_checked,
                adopted=False,
                full_results_count=0,
                duplicate_count=duplicate_count,
                scored_empty_count=scored_empty_count,
                gate_fail_reasons=gate_fail_reasons,
                validator_fail_reasons=validator_fail_reasons,
            )
            continue

        prefiltered.sort(key=lambda t: _critic_rank_score(t[1], critic_policy))
        selected = prefiltered[: max(1, k_full)]
        prefilter_map = {_candidate_hash(g.code): res for g, res in prefiltered}
        baseline_candidates = creator_candidates[:2]
        baseline_hashes = {_candidate_hash(c.code) for c in baseline_candidates}
        selected_hashes = {_candidate_hash(g.code) for g, _ in selected}
        for base in baseline_candidates:
            base_hash = _candidate_hash(base.code)
            if base_hash not in selected_hashes:
                base_res = prefilter_map.get(base_hash)
                if base_res:
                    selected.append((base, base_res))
        for g, pre_res in selected:
            append_blackboard(
                blackboard_path,
                {
                    "timestamp": now_ms(),
                    "agent_id": "critic",
                    "generation": r,
                    "candidate_hash": _candidate_hash(g.code),
                    "gate_ok": True,
                    "gate_reason": "",
                    "score_train": pre_res.train,
                    "score_holdout": pre_res.hold,
                    "score_stress": pre_res.stress,
                    "selected": True,
                    "note": "prefilter_selected",
                },
            )

        full_results: List[Tuple[Genome, EvalResult]] = []
        forced_eval = set(baseline_hashes)
        for g, _ in selected:
            if time.time() - critic_start > critic_slice and _candidate_hash(g.code) not in forced_eval:
                break
            refined = _critic_refine(round_rng, g, universe.meta, universe.library)
            for candidate in [g] + refined:
                if time.time() - critic_start > critic_slice and _candidate_hash(candidate.code) not in forced_eval:
                    break
                res = _evaluate_candidate(
                    candidate,
                    _merge_stress(fixed_batch, batch),
                    universe.eval_mode,
                    task.name,
                    extra_env=helper_env,
                    validator=validate_program if universe.eval_mode == "program" else validate_code,
                )
                if res.ok:
                    full_results.append((candidate, res))
                else:
                    if res.err:
                        if res.err.startswith("hard_gate:"):
                            gate_fail_reasons[res.err.split("hard_gate:", 1)[1]] += 1
                        else:
                            validator_fail_reasons[res.err] += 1
                append_blackboard(
                    blackboard_path,
                    {
                        "timestamp": now_ms(),
                        "agent_id": "critic",
                        "generation": r,
                        "candidate_hash": _candidate_hash(candidate.code),
                        "gate_ok": res.ok,
                        "gate_reason": "" if res.ok else (res.err or ""),
                        "score_train": res.train if res.ok else None,
                        "score_holdout": res.hold if res.ok else None,
                        "score_stress": res.stress if res.ok else None,
                        "selected": False,
                        "note": candidate.op_tag,
                    },
                )

        adopted = False
        full_results_count = len(full_results)
        if not full_results:
            scored_empty_count += 1
            reseed_templates = [_fallback_template_genome(round_rng, hint).statements]
            append_blackboard(
                blackboard_path,
                {
                    "timestamp": now_ms(),
                    "agent_id": "critic",
                    "generation": r,
                    "candidate_hash": "none",
                    "gate_ok": False,
                    "gate_reason": "scored_empty",
                    "score_train": None,
                    "score_holdout": None,
                    "score_stress": None,
                    "selected": False,
                    "note": "reseed",
                },
            )
            print("[DUO] No candidates survived full evaluation; reseeding templates.")
        else:
            full_results.sort(key=lambda t: t[1].score)
            best_g, best_res = full_results[0]
            if best_res.score < universe.best_score:
                adopted = True
                universe.best = best_g
                universe.best_score = best_res.score
                universe.best_train = best_res.train
                universe.best_hold = best_res.hold
                universe.best_stress = best_res.stress
                universe.best_test = best_res.test
                append_blackboard(
                    blackboard_path,
                    {
                        "timestamp": now_ms(),
                        "agent_id": "critic",
                        "generation": r,
                        "candidate_hash": _candidate_hash(best_g.code),
                        "gate_ok": True,
                        "gate_reason": "",
                        "score_train": best_res.train,
                        "score_holdout": best_res.hold,
                        "score_stress": best_res.stress,
                        "selected": True,
                        "note": "adopted",
                    },
                )
            universe.pool = [g for g, _ in full_results[: max(8, population // 4)]]
            if len(universe.pool) < population:
                universe.pool.extend([seed_genome(round_rng, hint) for _ in range(population - len(universe.pool))])

        gate_pass_rate = gate_pass / max(1, total_checked)
        creator_policy = _adjust_creator_policy(creator_policy, gate_pass_rate, gate_fail_reasons)
        _print_critic_summary(
            gate_pass=gate_pass,
            total_checked=total_checked,
            adopted=adopted,
            full_results_count=full_results_count,
            duplicate_count=duplicate_count,
            scored_empty_count=scored_empty_count,
            gate_fail_reasons=gate_fail_reasons,
            validator_fail_reasons=validator_fail_reasons,
        )

        gs = GlobalState(
            "RSI_EXTENDED_v2",
            now_ms(),
            now_ms(),
            seed,
            asdict(task),
            [universe.snapshot()],
            universe.uid,
            r + 1,
            mode=mode,
        )
        save_state(gs)

def run_rsi_loop(
    gens_per_round: int,
    rounds: int,
    levels: List[int],
    pop: int,
    n_univ: int,
    mode: str,
    freeze_eval: bool = True,
    meta_meta: bool = False,
    update_rule_rounds: int = 0,
):
    task = TaskSpec()
    seed = int(time.time()) % 100000
    if meta_meta:
        run_meta_meta(
            seed=seed,
            episodes=rounds,
            gens_per_episode=gens_per_round,
            pop=pop,
            n_univ=n_univ,
            freeze_eval=freeze_eval,
            state_dir=STATE_DIR,
            eval_every=1,
            few_shot_gens=max(3, gens_per_round // 2),
        )
        print(f"\n[RSI LOOP COMPLETE] {rounds} meta-meta rounds finished")
        return

    archive_path = STATE_DIR / "rsi_archive.json"
    archive = _load_rsi_archive(archive_path)
    if archive.get("current") and "genome" not in archive["current"]:
        archive = {"entries": [], "current": None, "consecutive": 0}
    fixed_batch = get_task_batch(task, seed, freeze_eval=True, gen=0)
    if fixed_batch is None:
        print("[RSI] No batch available; aborting.")
        return

    for r in range(rounds):
        print(f"\n{'='*60}\n[RSI ROUND {r+1}/{rounds}]\n{'='*60}")
        print(f"[EVOLVE] {gens_per_round} generations...")
        gs = run_multiverse(seed, task, gens_per_round, pop, n_univ, resume=(r > 0), mode=mode, freeze_eval=freeze_eval)
        best_snapshot = next((u for u in gs.universes if u.get("uid") == gs.selected_uid), None)
        best_data = (best_snapshot or {}).get("best")
        best_code = None
        if isinstance(best_data, dict):
            if mode == "learner":
                best_code = LearnerGenome(**best_data).code
            else:
                best_code = Genome(**best_data).code
        if best_code and best_code != "none":
            gate_ok, gate_reason = _hard_gate_ok(best_code, fixed_batch, mode, task.name)
            if not gate_ok:
                print(f"[RSI] Hard gate failed for best candidate ({gate_reason}); rejecting before scoring/autopatch.")
                archive["current"] = None
                archive["consecutive"] = 0
                archive["entries"] = []
                _save_rsi_archive(archive_path, archive)
                continue
        recent_scores = load_recent_scores(STATE_DIR / "run_log.jsonl", 5)
        forced_applied = False
        if is_300s_stagnation(recent_scores):
            print("[STAGNATION] 300s plateau detected for >=5 gens. Forcing L1/L3 autopatch.")
            forced = run_deep_autopatch([1, 3], candidates=4, apply=True, mode=mode)
            forced_applied = bool(forced.get("applied"))
            if forced_applied:
                print("[RSI] Self-modified via forced L1/L3 patch.")
            else:
                print("[STAGNATION] Forced patch rejected. Launching meta-meta acceleration.")
                run_meta_meta(
                    seed=seed,
                    episodes=1,
                    gens_per_episode=gens_per_round,
                    pop=pop,
                    n_univ=n_univ,
                    freeze_eval=freeze_eval,
                    state_dir=STATE_DIR,
                    eval_every=1,
                    few_shot_gens=max(3, gens_per_round // 2),
                )
                print("[STAGNATION] Meta-meta episode completed.")
        if not forced_applied:
            print(f"[AUTOPATCH] Trying L{levels}...")
            result = run_deep_autopatch(levels, candidates=4, apply=True, mode=mode)
            if result.get("applied"):
                print("[RSI] Self-modified! Reloading...")
        if update_rule_rounds > 0:
            print(f"[META] Running update-rule search for {update_rule_rounds} rounds...")
            run_update_rule_search(
                seed=seed + r * 127,
                rounds=update_rule_rounds,
                gens_per_round=max(3, gens_per_round // 2),
                pop=max(16, pop // 2),
                freeze_eval=freeze_eval,
                state_dir=STATE_DIR,
            )

    print(f"\n[RSI LOOP COMPLETE] {rounds} rounds finished")


# ---------------------------
# CLI Commands
# ---------------------------

def cmd_selftest(args):
    print("[selftest] Validating...")
    assert validate_expr("sin(x) + x*x")[0]
    assert not validate_expr("__import__('os')")[0]

    g = seed_genome(random.Random(42))
    t = TaskSpec()
    b = sample_batch(random.Random(42), t)
    assert b is not None
    r = evaluate(g, b, t.name)
    assert isinstance(r.score, float)

    hint = TaskDetective.detect_pattern(b)
    lg = seed_learner_genome(random.Random(42), hint)
    lr = evaluate_learner(lg, b, t.name)
    assert isinstance(lr.score, float)

    algo_code = "def run(inp):\n    return inp\n"
    assert validate_algo_program(algo_code)[0]

    print("[selftest] OK")
    return 0

def cmd_autopatch_probe(args):
    global STATE_DIR
    STATE_DIR = Path(args.state_dir)
    score = _autopatch_probe_score(
        mode=args.mode,
        task_name=args.task,
        seed=args.seed,
        generations=args.generations,
        population=args.population,
        universes=args.universes,
        freeze_eval=args.freeze_eval,
    )
    print(f"{score:.6f}")
    return 0

def cmd_evolve(args):
    global STATE_DIR
    STATE_DIR = Path(args.state_dir)
    resume = bool(args.resume) and (not args.fresh)
    mode = args.mode or ("algo" if args.task in ALGO_TASK_NAMES else "solver")
    run_multiverse(
        args.seed,
        TaskSpec(name=args.task),
        args.generations,
        args.population,
        args.universes,
        resume=resume,
        save_every=args.save_every,
        mode=mode,
        freeze_eval=args.freeze_eval,
    )
    print(f"\n[OK] State saved to {STATE_DIR / 'state.json'}")
    return 0

def cmd_learner_evolve(args):
    global STATE_DIR
    STATE_DIR = Path(args.state_dir)
    resume = bool(args.resume) and (not args.fresh)
    run_multiverse(
        args.seed,
        TaskSpec(name=args.task),
        args.generations,
        args.population,
        args.universes,
        resume=resume,
        save_every=args.save_every,
        mode="learner",
        freeze_eval=args.freeze_eval,
    )
    print(f"\n[OK] State saved to {STATE_DIR / 'state.json'}")
    return 0

def cmd_best(args):
    global STATE_DIR
    STATE_DIR = Path(args.state_dir)
    gs = load_state()
    if not gs:
        print("No state.")
        return 1
    u = next((s for s in gs.universes if s.get("uid") == gs.selected_uid), gs.universes[0] if gs.universes else {})
    best = u.get("best")
    if best:
        if gs.mode == "learner":
            g = LearnerGenome(**best)
        else:
            g = Genome(**best)
        print(g.code)
    print(f"Score: {u.get('best_score')} | Hold: {u.get('best_hold')} | Stress: {u.get('best_stress')} | Test: {u.get('best_test')}")
    print(f"Generations: {gs.generations_done}")
    return 0

def cmd_rsi_loop(args):
    global STATE_DIR
    STATE_DIR = Path(args.state_dir)
    levels = [int(l) for l in args.levels.split(",") if l.strip()]
    run_rsi_loop(
        args.generations,
        args.rounds,
        levels,
        args.population,
        args.universes,
        mode=args.mode,
        freeze_eval=args.freeze_eval,
        meta_meta=args.meta_meta,
        update_rule_rounds=args.update_rule_rounds,
    )
    return 0


def cmd_update_rule_search(args):
    global STATE_DIR
    STATE_DIR = Path(args.state_dir)
    run_update_rule_search(
        seed=args.seed,
        rounds=args.rounds,
        gens_per_round=args.generations,
        pop=args.population,
        freeze_eval=args.freeze_eval,
        state_dir=STATE_DIR,
    )
    return 0

def cmd_duo_loop(args):
    global STATE_DIR
    STATE_DIR = Path(args.state_dir)
    run_duo_loop(
        rounds=args.rounds,
        slice_seconds=args.slice_seconds,
        blackboard_path=Path(args.blackboard),
        k_full=args.k_full,
        seed=args.seed,
        mode=args.mode,
        freeze_eval=args.freeze_eval,
        population=args.population,
        max_candidates=args.max_candidates,
    )
    return 0

def cmd_meta_meta(args):
    global STATE_DIR
    STATE_DIR = Path(args.state_dir)
    run_meta_meta(
        seed=args.seed,
        episodes=args.episodes,
        gens_per_episode=args.gens_per_episode,
        pop=args.population,
        n_univ=args.universes,
        policy_pop=args.policy_pop,
        freeze_eval=args.freeze_eval,
        state_dir=STATE_DIR,
        eval_every=args.eval_every,
        few_shot_gens=args.few_shot_gens,
    )
    return 0

def cmd_task_switch(args):
    global STATE_DIR
    STATE_DIR = Path(args.state_dir)
    result = run_task_switch(
        seed=args.seed,
        task_a=TaskSpec(name=args.task_a),
        task_b=TaskSpec(name=args.task_b),
        gens_a=args.gens_a,
        gens_b=args.gens_b,
        pop=args.population,
        n_univ=args.universes,
        freeze_eval=args.freeze_eval,
        state_dir=STATE_DIR,
    )
    print(json.dumps(result, indent=2))
    return 0

def cmd_report(args):
    global STATE_DIR
    STATE_DIR = Path(args.state_dir)
    report = generate_report(STATE_DIR / "run_log.jsonl", args.few_shot_gens)
    print(json.dumps(report, indent=2))
    return 0


def cmd_transfer_bench(args):
    global STATE_DIR
    STATE_DIR = Path(args.state_dir)
    result = transfer_bench(args.task_from, args.task_to, args.budget, args.seed, freeze_eval=not args.no_freeze_eval)
    print(json.dumps(result, indent=2))
    return 0

def build_parser():
    p = argparse.ArgumentParser(prog="UNIFIED_RSI_EXTENDED", description="True RSI Engine with hard gates and rollback")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("selftest")
    s.set_defaults(fn=cmd_selftest)

    e = sub.add_parser("evolve")
    e.add_argument("--seed", type=int, default=1337)
    e.add_argument("--generations", type=int, default=80)
    e.add_argument("--population", type=int, default=128)
    e.add_argument("--universes", type=int, default=4)
    e.add_argument("--task", default="poly2")
    e.add_argument("--resume", action="store_true")
    e.add_argument("--fresh", action="store_true")
    e.add_argument("--save-every", type=int, default=5)
    e.add_argument("--state-dir", default=".rsi_state")
    e.add_argument("--freeze-eval", action=argparse.BooleanOptionalAction, default=True)
    e.add_argument("--mode", default="", choices=["", "solver", "algo"])
    e.set_defaults(fn=cmd_evolve)

    le = sub.add_parser("learner-evolve")
    le.add_argument("--seed", type=int, default=1337)
    le.add_argument("--generations", type=int, default=80)
    le.add_argument("--population", type=int, default=128)
    le.add_argument("--universes", type=int, default=4)
    le.add_argument("--task", default="poly2")
    le.add_argument("--resume", action="store_true")
    le.add_argument("--fresh", action="store_true")
    le.add_argument("--save-every", type=int, default=5)
    le.add_argument("--state-dir", default=".rsi_state")
    le.add_argument("--freeze-eval", action=argparse.BooleanOptionalAction, default=True)
    le.set_defaults(fn=cmd_learner_evolve)

    b = sub.add_parser("best")
    b.add_argument("--state-dir", default=".rsi_state")
    b.set_defaults(fn=cmd_best)

    r = sub.add_parser("rsi-loop")
    r.add_argument("--generations", type=int, default=50)
    r.add_argument("--rounds", type=int, default=5)
    r.add_argument("--population", type=int, default=64)
    r.add_argument("--universes", type=int, default=2)
    r.add_argument("--state-dir", default=".rsi_state")
    r.add_argument("--mode", default="solver", choices=["solver", "learner", "algo"])
    r.add_argument("--levels", default="1,2,3", help="Comma-separated autopatch levels (e.g., 1,3)")
    r.add_argument("--freeze-eval", action=argparse.BooleanOptionalAction, default=True)
    r.add_argument("--meta-meta", action="store_true", help="Run meta-meta loop instead of standard RSI rounds")
    r.add_argument("--update-rule-rounds", type=int, default=0, help="Rounds of update-rule search per RSI round")
    r.set_defaults(fn=cmd_rsi_loop)

    dl = sub.add_parser("duo-loop")
    dl.add_argument("--rounds", type=int, default=5)
    dl.add_argument("--slice-seconds", type=float, default=0.0)
    dl.add_argument("--blackboard", default=".rsi_blackboard.jsonl")
    dl.add_argument("--k-full", type=int, default=6)
    dl.add_argument("--seed", type=int, default=1337)
    dl.add_argument("--mode", default="solver", choices=["solver", "algo", "program"])
    dl.add_argument("--population", type=int, default=64)
    dl.add_argument("--max-candidates", type=int, default=512)
    dl.add_argument("--state-dir", default=".rsi_state")
    dl.add_argument("--freeze-eval", action=argparse.BooleanOptionalAction, default=True)
    dl.set_defaults(fn=cmd_duo_loop)

    ap = sub.add_parser("autopatch-probe")
    ap.add_argument("--mode", default="solver", choices=["solver", "learner", "algo"])
    ap.add_argument("--task", default="poly2")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--generations", type=int, default=6)
    ap.add_argument("--population", type=int, default=32)
    ap.add_argument("--universes", type=int, default=1)
    ap.add_argument("--state-dir", default=".rsi_state")
    ap.add_argument("--freeze-eval", action=argparse.BooleanOptionalAction, default=True)
    ap.set_defaults(fn=cmd_autopatch_probe)

    mm = sub.add_parser("meta-meta")
    mm.add_argument("--seed", type=int, default=1337)
    mm.add_argument("--episodes", type=int, default=20)
    mm.add_argument("--gens-per-episode", type=int, default=20)
    mm.add_argument("--population", type=int, default=64)
    mm.add_argument("--universes", type=int, default=2)
    mm.add_argument("--policy-pop", type=int, default=4)
    mm.add_argument("--state-dir", default=".rsi_state")
    mm.add_argument("--freeze-eval", action=argparse.BooleanOptionalAction, default=True)
    mm.add_argument("--eval-every", type=int, default=4)
    mm.add_argument("--few-shot-gens", type=int, default=10)
    mm.set_defaults(fn=cmd_meta_meta)

    ts = sub.add_parser("task-switch")
    ts.add_argument("--seed", type=int, default=1337)
    ts.add_argument("--task-a", default="poly2")
    ts.add_argument("--task-b", default="piecewise")
    ts.add_argument("--gens-a", type=int, default=10)
    ts.add_argument("--gens-b", type=int, default=10)
    ts.add_argument("--population", type=int, default=64)
    ts.add_argument("--universes", type=int, default=2)
    ts.add_argument("--state-dir", default=".rsi_state")
    ts.add_argument("--freeze-eval", action=argparse.BooleanOptionalAction, default=True)
    ts.set_defaults(fn=cmd_task_switch)

    tb = sub.add_parser("transfer-bench")
    tb.add_argument("--from", dest="task_from", required=True)
    tb.add_argument("--to", dest="task_to", required=True)
    tb.add_argument("--budget", type=int, default=12)
    tb.add_argument("--seed", type=int, default=1337)
    tb.add_argument("--state-dir", default=".rsi_state")
    tb.add_argument("--no-freeze-eval", action="store_true")
    tb.set_defaults(fn=cmd_transfer_bench)

    rp = sub.add_parser("report")
    rp.add_argument("--state-dir", default=".rsi_state")
    rp.add_argument("--few-shot-gens", type=int, default=10)
    rp.set_defaults(fn=cmd_report)

    inv = sub.add_parser("invention")
    inv.add_argument("--seed", type=int, default=0)
    inv.add_argument("--iterations", type=int, default=6)
    inv.set_defaults(fn=cmd_invention)

    ur = sub.add_parser("update-rule")
    ur.add_argument("--seed", type=int, default=1337)
    ur.add_argument("--rounds", type=int, default=4)
    ur.add_argument("--generations", type=int, default=6)
    ur.add_argument("--population", type=int, default=32)
    ur.add_argument("--state-dir", default=".rsi_state")
    ur.add_argument("--freeze-eval", action=argparse.BooleanOptionalAction, default=True)
    ur.set_defaults(fn=cmd_update_rule_search)

    return p

def main():
    parser = build_parser()
    if len(sys.argv) == 1:
        sys.argv.append("selftest")
    args = parser.parse_args()
    return args.fn(args)

if __name__ == "__main__":
    raise SystemExit(main())
