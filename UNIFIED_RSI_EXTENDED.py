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
  python UNIFIED_RSI_EXTENDED.py learner-evolve --fresh --generations 100
  python UNIFIED_RSI_EXTENDED.py autopatch --levels 0,1,3 --apply
  python UNIFIED_RSI_EXTENDED.py rsi-loop --generations 50 --rounds 10
  python UNIFIED_RSI_EXTENDED.py rsi-loop --generations 20 --rounds 5 --mode learner
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
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Set


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
    # list helpers
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

        exec(compile(tree, "<lgp>", "exec"), env)
        if "run" not in env:
            return float("nan")
        return env["run"](x)
    except StepLimitExceeded:
        return float("nan")
    except Exception:
        return float("nan")


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
class TaskSpec:
    name: str = "poly2"
    x_min: float = -3.0
    x_max: float = 3.0
    n_train: int = 96
    n_hold: int = 96
    noise: float = 0.01
    stress_mult: float = 3.0
    target_code: Optional[str] = None


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
    "sinmix": lambda x: math.sin(x) + 0.3 * math.cos(2 * x),
    "absline": lambda x: abs(x) + 0.2 * x,
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
        return Batch(x_tr, y_tr, x_ho, y_ho, x_st, y_st)

    if t.name in ("sort", "reverse", "filter", "max"):
        x_tr = gen_lists(t.n_train, t.x_min, t.x_max)
        x_ho = gen_lists(t.n_hold, t.x_min + 2, t.x_max + 2)
        x_st = gen_lists(t.n_hold, t.x_max + 5, t.x_max + 10)
        y_tr = [f(x) for x in x_tr]
        y_ho = [f(x) for x in x_ho]
        y_st = [f(x) for x in x_st]
        return Batch(x_tr, y_tr, x_ho, y_ho, x_st, y_st)

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
        return Batch(x_tr, y_tr, x_ho, y_ho, x_st, y_st)

    # numeric regression tasks
    xs = lambda n, a, b: [a + (b - a) * rng.random() for _ in range(n)]
    ys = lambda xv, n: [f(x) + rng.gauss(0, n) if n > 0 else f(x) for x in xv]
    half = 0.5 * (t.x_max - t.x_min)
    mid = 0.5 * (t.x_min + t.x_max)
    x_tr = xs(t.n_train, t.x_min, t.x_max)
    x_ho = xs(t.n_hold, t.x_min, t.x_max)
    x_st = xs(t.n_hold, mid - half * t.stress_mult, mid + half * t.stress_mult)
    return Batch(x_tr, ys(x_tr, t.noise), x_ho, ys(x_ho, t.noise), x_st, ys(x_st, t.noise * t.stress_mult))


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
    nodes: int
    score: float
    err: Optional[str] = None


SCORE_W_HOLD = 0.49
SCORE_W_STRESS = 0.28
SCORE_W_TRAIN = 0.05


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


def mse_exec(code: str, xs: List[Any], ys: List[Any], task_name: str = "", extra_env: Optional[Dict[str, Any]] = None) -> Tuple[bool, float, str]:
    ok, err = validate_code(code)
    if not ok:
        return (False, float("inf"), err)
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


def evaluate(g: Genome, b: Batch, task_name: str, lam: float = 0.0001, extra_env: Optional[Dict[str, Any]] = None) -> EvalResult:
    code = g.code
    ok1, tr, e1 = mse_exec(code, b.x_tr, b.y_tr, task_name, extra_env=extra_env)
    ok2, ho, e2 = mse_exec(code, b.x_ho, b.y_ho, task_name, extra_env=extra_env)
    ok3, st, e3 = mse_exec(code, b.x_st, b.y_st, task_name, extra_env=extra_env)
    ok = ok1 and ok2 and ok3 and all(math.isfinite(v) for v in (tr, ho, st))
    nodes = node_count(code)
    score = SCORE_W_HOLD * ho + SCORE_W_STRESS * st + SCORE_W_TRAIN * tr + lam * nodes
    err = e1 or e2 or e3
    return EvalResult(ok, tr, ho, st, nodes, score, err or None)


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
        return EvalResult(False, float("inf"), float("inf"), float("inf"), 0, float("inf"), "load_failed")
    required = ["init_mem", "encode", "predict", "update", "objective"]
    if not all(name in env and callable(env[name]) for name in required):
        return EvalResult(False, float("inf"), float("inf"), float("inf"), 0, float("inf"), "missing_funcs")

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
        nodes = node_count(learner.code)
        obj = objective(train, hold, stress, nodes)
        if not isinstance(obj, (int, float)) or not math.isfinite(obj):
            obj = SCORE_W_HOLD * hold + SCORE_W_STRESS * stress + SCORE_W_TRAIN * train
        score = float(obj) + lam * nodes
        ok = all(math.isfinite(v) for v in (train, hold, stress, score))
        return EvalResult(ok, train, hold, stress, nodes, score, None if ok else "nan")
    except Exception as exc:
        return EvalResult(False, float("inf"), float("inf"), float("inf"), 0, float("inf"), str(exc))


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

    @staticmethod
    def from_snapshot(s: Dict) -> "FunctionLibrary":
        lib = FunctionLibrary()
        for fd in s.get("funcs", []):
            lib.funcs[fd["name"]] = LearnedFunc(**fd)
        return lib


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
    best: Optional[Genome] = None
    best_score: float = float("inf")
    best_hold: float = float("inf")
    best_stress: float = float("inf")
    history: List[Dict] = field(default_factory=list)

    def step(self, gen: int, task: TaskSpec, pop_size: int) -> Dict:
        rng = random.Random(self.seed + gen * 1009)
        batch = sample_batch(rng, task)
        if batch is None:
            self.pool = [seed_genome(rng) for _ in range(pop_size)]
            return {"gen": gen, "accepted": False, "reason": "no_batch"}

        helper_env = self.library.get_helpers()

        scored: List[Tuple[Genome, EvalResult]] = []
        all_results: List[Tuple[Genome, EvalResult]] = []
        for g in self.pool:
            res = evaluate(g, batch, task.name, self.meta.complexity_lambda, extra_env=helper_env)
            all_results.append((g, res))
            if res.ok:
                scored.append((g, res))

        MetaCognitiveEngine.analyze_execution(all_results, self.meta)

        if not scored:
            hint = TaskDetective.detect_pattern(batch)
            self.pool = [seed_genome(rng, hint) for _ in range(pop_size)]
            return {"gen": gen, "accepted": False, "reason": "reseed"}

        scored.sort(key=lambda t: t[1].score)

        # MAP-Elites add
        best_g0, best_res0 = scored[0]
        MAP_ELITES.add(best_g0, best_res0.score)

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

            candidates.append(Genome(statements=new_stmts, parents=[parent.gid], op_tag=op_tag))

        # surrogate ranking
        with_pred = [(c, SURROGATE.predict(c.code)) for c in candidates]
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
        accepted = best_res.score < self.best_score - 1e-9
        if accepted:
            self.best = best_g
            self.best_score = best_res.score
            self.best_hold = best_res.hold
            self.best_stress = best_res.stress

        op_used = best_g.op_tag.split(":")[1].split("|")[0] if ":" in best_g.op_tag else "unknown"
        self.meta.update(op_used, self.best_score - old_score, accepted)

        log = {
            "gen": gen,
            "accepted": accepted,
            "score": self.best_score,
            "hold": self.best_hold,
            "stress": self.best_stress,
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
            "pool": [asdict(g) for g in self.pool[:20]],
            "library": self.library.snapshot(),
            "history": self.history[-50:],
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
        u.best_hold = s.get("best_hold", float("inf"))
        u.best_stress = s.get("best_stress", float("inf"))
        u.history = s.get("history", [])
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
    history: List[Dict] = field(default_factory=list)

    def step(self, gen: int, task: TaskSpec, pop_size: int) -> Dict:
        rng = random.Random(self.seed + gen * 1009)
        batch = sample_batch(rng, task)
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

        op_used = best_g.op_tag.split(":")[1].split("|")[0] if ":" in best_g.op_tag else "unknown"
        self.meta.update(op_used, self.best_score - old_score, accepted)

        log = {
            "gen": gen,
            "accepted": accepted,
            "score": self.best_score,
            "hold": self.best_hold,
            "stress": self.best_stress,
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
) -> GlobalState:
    safe_mkdir(STATE_DIR)

    if resume and (gs0 := load_state()):
        mode = gs0.mode
        if mode == "learner":
            us = [UniverseLearner.from_snapshot(s) for s in gs0.universes]
        else:
            us = [Universe.from_snapshot(s) for s in gs0.universes]
        start = gs0.generations_done
    else:
        b0 = sample_batch(random.Random(seed), task)
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
            us = [
                Universe(
                    uid=i,
                    seed=seed + i * 9973,
                    meta=MetaState(),
                    pool=[seed_genome(random.Random(seed + i), hint) for _ in range(pop)],
                    library=FunctionLibrary(),
                )
                for i in range(n_univ)
            ]
        start = 0

    for gen in range(start, start + gens):
        for u in us:
            u.step(gen, task, pop)

        us.sort(key=lambda u: u.best_score)
        best = us[0]
        print(
            f"[Gen {gen + 1:4d}] Score: {best.best_score:.4f} | Hold: {best.best_hold:.4f} | Stress: {best.best_stress:.4f} | "
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

def run_rsi_loop(gens_per_round: int, rounds: int, levels: List[int], pop: int, n_univ: int, mode: str):
    task = TaskSpec()
    seed = int(time.time()) % 100000
    for r in range(rounds):
        print(f"\n{'='*60}\n[RSI ROUND {r+1}/{rounds}]\n{'='*60}")
        print(f"[EVOLVE] {gens_per_round} generations...")
        run_multiverse(seed, task, gens_per_round, pop, n_univ, resume=(r > 0), mode=mode)
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

    print("[selftest] OK")
    return 0

def cmd_evolve(args):
    global STATE_DIR
    STATE_DIR = Path(args.state_dir)
    resume = bool(args.resume) and (not args.fresh)
    run_multiverse(args.seed, TaskSpec(name=args.task), args.generations, args.population, args.universes, resume=resume, save_every=args.save_every, mode="solver")
    print(f"\n[OK] State saved to {STATE_DIR / 'state.json'}")
    return 0

def cmd_learner_evolve(args):
    global STATE_DIR
    STATE_DIR = Path(args.state_dir)
    resume = bool(args.resume) and (not args.fresh)
    run_multiverse(args.seed, TaskSpec(name=args.task), args.generations, args.population, args.universes, resume=resume, save_every=args.save_every, mode="learner")
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
    print(f"Score: {u.get('best_score')} | Hold: {u.get('best_hold')} | Stress: {u.get('best_stress')}")
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
    run_rsi_loop(args.generations, args.rounds, levels, args.population, args.universes, mode=args.mode)
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
    le.set_defaults(fn=cmd_learner_evolve)

    b = sub.add_parser("best")
    b.add_argument("--state-dir", default=".rsi_state")
    b.set_defaults(fn=cmd_best)

    a = sub.add_parser("autopatch")
    a.add_argument("--levels", default="0,1,3")
    a.add_argument("--candidates", type=int, default=4)
    a.add_argument("--apply", action="store_true")
    a.add_argument("--state-dir", default=".rsi_state")
    a.add_argument("--mode", default="", choices=["", "solver", "learner"])
    a.set_defaults(fn=cmd_autopatch)

    r = sub.add_parser("rsi-loop")
    r.add_argument("--generations", type=int, default=50)
    r.add_argument("--rounds", type=int, default=5)
    r.add_argument("--levels", default="0,1,3")
    r.add_argument("--population", type=int, default=64)
    r.add_argument("--universes", type=int, default=2)
    r.add_argument("--state-dir", default=".rsi_state")
    r.add_argument("--mode", default="solver", choices=["solver", "learner"])
    r.set_defaults(fn=cmd_rsi_loop)

    return p

def main():
    parser = build_parser()
    args = parser.parse_args()
    return args.fn(args)

if __name__ == "__main__":
    raise SystemExit(main())
