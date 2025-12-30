"""
UNIFIED_RSI_EXTENDED.py

True RSI (Recursive Self-Improvement) Engine - BETA (Executable)
================================================================

RSI Levels:
- L0: Hyperparameter tuning
- L1: Operator weight adaptation (source-patchable via OP_WEIGHT_INIT)
- L2: Add/remove mutation operators (runtime library evolution)
- L3: Modify evaluation function weights
- L4: Persist learned operators into source (optional; markers supported)
- L5: Modify self-modification logic (simple source edits)

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
  python UNIFIED_RSI_EXTENDED.py autopatch --levels 0,1,3 --apply
  python UNIFIED_RSI_EXTENDED.py rsi-loop --generations 50 --rounds 10
  python UNIFIED_RSI_EXTENDED.py rsi-loop --generations 20 --rounds 5 --mode learner

CHANGELOG
---------
L0: Solver supports expression genomes and strict program-mode genomes (Assign/If/Return only).
L1: RuleDSL controls mutation/crossover/novelty/acceptance/curriculum knobs per generation.
L2: Meta-meta loop proposes RuleDSL patches and accepts only when meta-test transfer improves.
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
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Set, Union


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
        if isinstance(node, ast.Call):
            # forbid attribute calls (e.g., os.system)
            if not isinstance(node.func, ast.Name):
                self.ok, self.err = (False, "Forbidden call form (non-Name callee)")
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
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                self.ok, self.err = (False, "Forbidden call form (non-Name callee)")
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
    """Algo-mode validator with bounded structure and no attribute access."""

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
            self.ok, self.err = (False, "Forbidden attribute access")
            return
        if isinstance(node, ast.Name):
            if node.id.startswith("__") or node.id in ("open", "eval", "exec", "compile", "__import__", "globals", "locals"):
                self.ok, self.err = (False, f"Forbidden name: {node.id}")
                return
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                self.ok, self.err = (False, "Forbidden call form (non-Name callee)")
                return
        super().generic_visit(node)


def algo_program_limits_ok(
    code: str,
    max_nodes: int = 280,
    max_depth: int = 24,
    max_funcs: int = 4,
    max_locals: int = 24,
    max_consts: int = 64,
    max_subscripts: int = 32,
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
        ast.Name, ast.Load,
        ast.Constant,
        ast.List, ast.Tuple, ast.Dict,
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
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                self.ok, self.err = (False, "Forbidden call form (non-Name callee)")
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
        elif self.name in ALGO_TASK_NAMES:
            family = "algo"
        elif self.name.startswith("arc_"):
            family = "arc"
        self.descriptor = TaskDescriptor(
            name=self.name,
            family=family,
            input_kind="list" if family in ("list", "algo") else ("grid" if family == "arc" else "scalar"),
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

def sample_batch(rng: random.Random, t: TaskSpec) -> Optional[Batch]:
    # function target
    if t.target_code:
        f = lambda x: safe_exec(t.target_code, x)
    elif t.name in ("sort", "reverse", "filter", "max"):
        f = TARGET_FNS.get(t.name) or (lambda x: sorted(x))
    else:
        f = TARGET_FNS.get(t.name, lambda x: x)

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
        TaskSpec(name="piecewise", x_min=-4.0, x_max=4.0, n_train=96, n_hold=64, n_test=64, noise=0.01),
        TaskSpec(name="rational", x_min=-5.0, x_max=5.0, n_train=96, n_hold=64, n_test=64, noise=0.02),
        TaskSpec(name="switching", x_min=-3.0, x_max=3.0, n_train=96, n_hold=64, n_test=64, noise=0.0),
        TaskSpec(name="classification", x_min=-4.0, x_max=4.0, n_train=96, n_hold=64, n_test=64, noise=0.0),
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


SCORE_W_HOLD = 0.6
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
    ok = ok1 and ok2 and ok3 and ok4
    nodes = node_count(code)
    step_penalty = 0.0001 * (tr_steps + ho_steps + st_steps + te_steps)
    timeout_penalty = 0.5 * (tr_timeout + ho_timeout + st_timeout + te_timeout)
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
class MetaState:
    op_weights: Dict[str, float] = field(default_factory=lambda: dict(OP_WEIGHT_INIT))
    mutation_rate: float = 0.8863
    crossover_rate: float = 0.1971
    complexity_lambda: float = 0.0001
    epsilon_explore: float = 0.4213
    adapt_steps: int = 8
    stuck_counter: int = 0
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
            reward = max(0.0, -delta) if accepted else -0.1
            self.op_weights[op] = clamp(self.op_weights[op] + 0.1 * reward, 0.1, 5.0)
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

    if resume and (gs0 := load_state()):
        mode = gs0.mode
        if mode == "learner":
            us = [UniverseLearner.from_snapshot(s) for s in gs0.universes]
        else:
            us = [Universe.from_snapshot(s) for s in gs0.universes]
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
                    meta=MetaState(),
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
                    meta=MetaState(),
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
    universes = [
        Universe(
            uid=i,
            seed=seed + i * 9973,
            meta=MetaState(),
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
                hist, _ = run_policy_episode(
                    seed + episode * 73,
                    task_test,
                    policy,
                    few_shot_gens,
                    pop,
                    n_univ,
                    freeze_eval,
                    archive,
                    logger,
                    mode="meta-test",
                    update_archive=False,
                )
                transfer_scores.append(compute_transfer_metrics(hist, window=few_shot_gens)["auc"])
            if transfer_scores:
                policy_scores[policy.pid] = sum(transfer_scores) / len(transfer_scores)
            policies.sort(key=lambda p: policy_scores.get(p.pid, float("inf")))
            best_policy = policies[0]
            policies = [best_policy] + [best_policy.mutate(rng, scale=0.05) for _ in range(policy_pop - 1)]


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
# Self-modification (AutoPatch)
# ---------------------------

PATCH_LEVELS = {0: "hyperparameter", 1: "op_weight", 2: "operator_toggle", 3: "eval_weight", 4: "operator_persist", 5: "meta_logic"}

@dataclass
class PatchPlan:
    level: int
    patch_id: str
    title: str
    rationale: str
    new_source: str
    diff: str

def _read_self() -> str:
    return Path(__file__).read_text(encoding="utf-8")

def _patch_dataclass(src: str, cls: str, field_name: str, val: Any) -> Tuple[bool, str]:
    try:
        mod = ast.parse(src)
        patched = False
        for node in ast.walk(mod):
            if isinstance(node, ast.ClassDef) and node.name == cls:
                for stmt in node.body:
                    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.target.id == field_name:
                        stmt.value = ast.Constant(value=val)
                        patched = True
        if not patched:
            return (False, src)
        ast.fix_missing_locations(mod)
        return (True, ast.unparse(mod))
    except Exception:
        return (False, src)

def _patch_global_const(src: str, name: str, val: float) -> Tuple[bool, str]:
    pattern = f"^({re.escape(name)}\\s*=\\s*)[\\d.eE+-]+"
    new_src, n = re.subn(pattern, f"\\g<1>{val}", src, flags=re.MULTILINE)
    return (n > 0, new_src)

def _patch_dict_const(src: str, dict_name: str, key: str, val: float) -> Tuple[bool, str]:
    try:
        mod = ast.parse(src)
        patched = False
        for node in ast.walk(mod):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id == dict_name and isinstance(node.value, ast.Dict):
                        # find key
                        for i, k in enumerate(node.value.keys):
                            if isinstance(k, ast.Constant) and k.value == key:
                                node.value.values[i] = ast.Constant(value=float(val))
                                patched = True
        if not patched:
            return (False, src)
        ast.fix_missing_locations(mod)
        return (True, ast.unparse(mod))
    except Exception:
        return (False, src)

def _rewrite_operators_block(src: str, new_lib: Dict) -> str:
    pattern = r"(# @@OPERATORS_LIB_START@@\s*\nOPERATORS_LIB:\s*Dict\[str,\s*Dict\]\s*=\s*)(\{.*?\})(\s*\n# @@OPERATORS_LIB_END@@)"
    match = re.search(pattern, src, flags=re.DOTALL)
    if not match:
        return src
    prefix, _, suffix = match.group(1), match.group(2), match.group(3)
    lines = ["{"]
    for name, spec in new_lib.items():
        lines.append(f'    "{name}": {json.dumps(spec)},')
    lines.append("}")
    new_dict = "\n".join(lines)
    return src[: match.start()] + prefix + new_dict + suffix + src[match.end():]

def propose_patches(gs: GlobalState, levels: List[int]) -> List[PatchPlan]:
    src = _read_self()
    plans: List[PatchPlan] = []
    rng = random.Random(gs.updated_ms)

    best_u = next((s for s in gs.universes if s.get("uid") == gs.selected_uid), gs.universes[0] if gs.universes else {})
    meta = best_u.get("meta", {})

    # L0: tune hyperparams (dataclass literal fields)
    if 0 in levels:
        for cls, field_name, base in [
            ("MetaState", "mutation_rate", 0.65),
            ("MetaState", "crossover_rate", 0.2),
            ("MetaState", "epsilon_explore", 0.15),
        ]:
            cur = meta.get(field_name, base)
            new_val = round(clamp(cur * rng.uniform(0.85, 1.15), 0.1, 0.95), 4)
            ok, new_src = _patch_dataclass(src, cls, field_name, new_val)
            if ok and new_src != src:
                plans.append(PatchPlan(
                    0,
                    sha256(f"{cls}.{field_name}={new_val}")[:8],
                    f"L0: {cls}.{field_name} -> {new_val}",
                    "Adaptive tuning",
                    new_src,
                    unified_diff(src, new_src, "script.py"),
                ))

    # L1: patch OP_WEIGHT_INIT dict entries (source-patchable now)
    if 1 in levels:
        op_w = meta.get("op_weights", {})
        # pick a few ops to perturb
        if isinstance(op_w, dict) and op_w:
            ops = list(op_w.items())[: min(4, len(op_w))]
        else:
            ops = list(OP_WEIGHT_INIT.items())[:4]
        new_src = src
        changed = False
        for op, w in ops:
            new_w = round(clamp(float(w) * rng.uniform(0.9, 1.1), 0.1, 5.0), 3)
            ok, new_src2 = _patch_dict_const(new_src, "OP_WEIGHT_INIT", op, new_w)
            if ok and new_src2 != new_src:
                changed = True
                new_src = new_src2
        if changed and new_src != src:
            plans.append(PatchPlan(
                1,
                sha256("L1:" + str(rng.random()))[:8],
                "L1: OP_WEIGHT_INIT perturb",
                "Operator-weight prior update",
                new_src,
                unified_diff(src, new_src, "script.py"),
            ))

    # L3: rebalance eval weights
    if 3 in levels:
        for name, base in [("SCORE_W_HOLD", 0.6), ("SCORE_W_STRESS", 0.35), ("SCORE_W_TRAIN", 0.05)]:
            new_val = round(clamp(base * rng.uniform(0.8, 1.2), 0.05, 0.9), 2)
            ok, new_src = _patch_global_const(src, name, new_val)
            if ok and new_src != src:
                plans.append(PatchPlan(
                    3,
                    sha256(f"{name}={new_val}")[:8],
                    f"L3: {name} -> {new_val}",
                    "Eval rebalancing",
                    new_src,
                    unified_diff(src, new_src, "script.py"),
                ))

    # L4: persist OPERATORS_LIB into source (optional)
    if 4 in levels and OPERATORS_LIB:
        new_src = _rewrite_operators_block(src, OPERATORS_LIB)
        if new_src != src:
            plans.append(PatchPlan(
                4,
                sha256(str(OPERATORS_LIB))[:8],
                f"L4: Persist {len(OPERATORS_LIB)} learned operators",
                f"Operators: {list(OPERATORS_LIB.keys())[:5]}",
                new_src,
                unified_diff(src, new_src, "script.py"),
            ))

    # L5: simple meta-logic edits (conservative)
    if 5 in levels:
        elite_mods = [
            ("max(4, pop_size // 10)", "max(4, pop_size // 8)"),
            ("max(4, pop_size // 8)", "max(4, pop_size // 12)"),
            ("max(4, pop_size // 12)", "max(4, pop_size // 10)"),
        ]
        for old_pat, new_pat in elite_mods:
            if old_pat in src:
                new_src = src.replace(old_pat, new_pat, 1)
                plans.append(PatchPlan(
                    5,
                    sha256(new_pat)[:8],
                    "L5: elite ratio change",
                    "Meta: selection pressure",
                    new_src,
                    unified_diff(src, new_src, "script.py"),
                ))
                break

    rng.shuffle(plans)
    return plans[:8]

def _probe_tasks(task_name: str) -> List[str]:
    if task_name == "poly2":
        return ["poly2", "poly3"]
    if task_name in ("sort", "reverse", "filter", "max"):
        return [task_name, "reverse" if task_name != "reverse" else "sort"]
    return [task_name]

def probe_run(script: Path, mode: str, task_name: str, gens: int = 5, pop: int = 16) -> float:
    """PHASE D: probe on multiple seeds/tasks to avoid overfitting."""
    seeds = [11, 23, 37]
    tasks = _probe_tasks(task_name)
    scores: List[float] = []
    cmd_name = "learner-evolve" if mode == "learner" else "evolve"

    with tempfile.TemporaryDirectory() as td:
        for seed in seeds:
            for task in tasks:
                try:
                    proc = subprocess.run(
                        [sys.executable, str(script), cmd_name, "--fresh", "--generations", str(gens),
                         "--population", str(pop), "--universes", "1", "--state-dir", td, "--seed", str(seed), "--task", task],
                        capture_output=True,
                        text=True,
                        timeout=120,
                    )
                    for line in reversed(proc.stdout.splitlines()):
                        if "Score:" in line:
                            m = re.search(r"Score:\s*([\d.]+)", line)
                            if m:
                                scores.append(float(m.group(1)))
                                break
                except Exception:
                    scores.append(float("inf"))
    if not scores:
        return float("inf")
    return sum(scores) / len(scores)

def run_deep_autopatch(levels: List[int], candidates: int = 4, apply: bool = False, mode: str = "solver") -> Dict:
    gs = load_state()
    if not gs:
        return {"error": "No state. Run evolve first."}

    mode = mode or gs.mode
    task_name = gs.task.get("name", "poly2") if isinstance(gs.task, dict) else "poly2"

    script = Path(__file__).resolve()
    baseline = probe_run(script, mode, task_name)
    print(f"[AUTOPATCH L{levels}] Baseline: {baseline:.4f}")

    plans = propose_patches(gs, levels)[:candidates]
    if not plans:
        return {"error": "No patches generated"}

    results = []
    best_plan, best_score = (None, baseline)

    for p in plans:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(p.new_source)
            tmp = Path(f.name)
        try:
            score = probe_run(tmp, mode, task_name)
            improved = score < baseline - 1e-6
            results.append({"level": p.level, "id": p.patch_id, "title": p.title, "score": score, "improved": improved})
            print(f"[L{p.level}] {p.patch_id}: {p.title} -> {score:.4f} {'OK' if improved else 'FAIL'}")
            if improved and score < best_score:
                best_score, best_plan = score, p
        finally:
            tmp.unlink(missing_ok=True)

    if best_plan and apply:
        backup = script.with_suffix(".bak")
        if not backup.exists():
            backup.write_text(script.read_text(encoding="utf-8"), encoding="utf-8")
        script.write_text(best_plan.new_source, encoding="utf-8")
        print(f"[OK] Applied L{best_plan.level} patch: {best_plan.title}")
        return {"applied": best_plan.patch_id, "level": best_plan.level, "score": best_score, "results": results}

    if best_plan:
        out = STATE_DIR / "patched.py"
        out.write_text(best_plan.new_source, encoding="utf-8")
        print(f"[OK] Best patch saved to {out}")
        return {"best": best_plan.patch_id, "score": best_score, "file": str(out), "results": results}

    return {"improved": False, "baseline": baseline, "results": results}

def run_rsi_loop(gens_per_round: int, rounds: int, levels: List[int], pop: int, n_univ: int, mode: str, freeze_eval: bool = True):
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

    for r in range(rounds):
        print(f"\n{'='*60}\n[RSI ROUND {r+1}/{rounds}]\n{'='*60}")
        print(f"[EVOLVE] {gens_per_round} generations...")
        run_multiverse(seed, task, gens_per_round, pop, n_univ, resume=(r > 0), mode=mode, freeze_eval=freeze_eval)
        print(f"[AUTOPATCH] Trying L{levels}...")
        result = run_deep_autopatch(levels, candidates=4, apply=True, mode=mode)
        if result.get("applied"):
            print("[RSI] Self-modified! Reloading...")

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

def cmd_autopatch(args):
    global STATE_DIR
    STATE_DIR = Path(args.state_dir)
    levels = [int(l) for l in args.levels.split(",") if l.strip()]
    mode = args.mode or ""
    result = run_deep_autopatch(levels, args.candidates, args.apply, mode=mode)
    print(json.dumps(result, indent=2, default=str))
    return 0

def cmd_rsi_loop(args):
    global STATE_DIR
    STATE_DIR = Path(args.state_dir)
    levels = [int(l) for l in args.levels.split(",") if l.strip()]
    run_rsi_loop(args.generations, args.rounds, levels, args.population, args.universes, mode=args.mode, freeze_eval=args.freeze_eval)
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
    p = argparse.ArgumentParser(prog="UNIFIED_RSI_EXTENDED", description="True RSI Engine with L0-L5 Self-Modification")
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

    a = sub.add_parser("autopatch")
    a.add_argument("--levels", default="0,1,3")
    a.add_argument("--candidates", type=int, default=4)
    a.add_argument("--apply", action="store_true")
    a.add_argument("--state-dir", default=".rsi_state")
    a.add_argument("--mode", default="", choices=["", "solver", "learner", "algo"])
    a.set_defaults(fn=cmd_autopatch)

    r = sub.add_parser("rsi-loop")
    r.add_argument("--generations", type=int, default=50)
    r.add_argument("--rounds", type=int, default=5)
    r.add_argument("--levels", default="0,1,3")
    r.add_argument("--population", type=int, default=64)
    r.add_argument("--universes", type=int, default=2)
    r.add_argument("--state-dir", default=".rsi_state")
    r.add_argument("--mode", default="solver", choices=["solver", "learner", "algo"])
    r.add_argument("--freeze-eval", action=argparse.BooleanOptionalAction, default=True)
    r.set_defaults(fn=cmd_rsi_loop)

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

    return p

def main():
    parser = build_parser()
    if len(sys.argv) == 1:
        sys.argv.append("selftest")
    args = parser.parse_args()
    return args.fn(args)

if __name__ == "__main__":
    raise SystemExit(main())
