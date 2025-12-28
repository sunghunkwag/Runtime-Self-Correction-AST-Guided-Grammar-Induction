"""UNIFIED_RSI_EXTENDED.py

True RSI (Recursive Self-Improvement) Engine - BETA
====================================================
⚠️ BETA TESTING - Features under active development
RSI Levels:
- L0: Hyperparameter tuning
- L1: Operator weight adaptation  
- L2: Add/remove mutation operators
- L3: Modify evaluation function
- L4: Synthesize new operators
- L5: Modify self-modification logic

CLI:
  python UNIFIED_RSI_EXTENDED.py selftest
  python UNIFIED_RSI_EXTENDED.py evolve --fresh --generations 100
  python UNIFIED_RSI_EXTENDED.py autopatch --levels 0,1,2,3 --apply
  python UNIFIED_RSI_EXTENDED.py rsi-loop --generations 50 --rounds 10
"""
from __future__ import annotations
import argparse, ast, dataclasses, difflib, hashlib, json, math, os, random
import re, subprocess, sys, tempfile, time, textwrap
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Set

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
        return json.loads(p.read_text(encoding='utf-8'))
    except:
        return {}

def write_json(p: Path, obj: Any, indent: int=2):
    safe_mkdir(p.parent)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=indent, default=str), encoding='utf-8')

def unified_diff(old: str, new: str, name: str) -> str:
    return ''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), fromfile=name, tofile=name))
SAFE_FUNCS: Dict[str, Callable] = {
    'sin': math.sin, 'cos': math.cos, 'tan': math.tan, 'exp': math.exp, 'tanh': math.tanh, 
    'abs': abs, 'sqrt': lambda x: math.sqrt(abs(x) + 1e-12), 'log': lambda x: math.log(abs(x) + 1e-12), 
    'pow2': lambda x: x * x, 'sigmoid': lambda x: 1.0 / (1.0 + math.exp(-clamp(x, -500, 500))), 
    'gamma': lambda x: math.gamma(abs(x) + 1e-09) if abs(x) < 170 else float('inf'), 
    'erf': math.erf, 'ceil': math.ceil, 'floor': math.floor, 'sign': lambda x: math.copysign(1.0, x),
    'sorted': sorted, 'reversed': reversed, 'max': max, 'min': min, 'sum': sum, 'len': len, 'list': list
}
# [NEW] Phase 4: Bayesian Grammar Weights (EDA-Learned)
GRAMMAR_PROBS: Dict[str, float] = {k: 1.0 for k in SAFE_FUNCS} # Default uniform
GRAMMAR_PROBS.update({'binop': 2.0, 'call': 1.0, 'const': 1.0, 'var': 2.0})

SAFE_BUILTINS = {'abs': abs, 'min': min, 'max': max, 'float': float, 'int': int, 'len': len, 'range': range, 'list': list, 'sorted': sorted, 'reversed': reversed, 'sum': sum} # [NEW] Algorithmic Builtins
SAFE_VARS = {'x'}

# [NEW] Phase 4: ARC-Lite DSL (Grid Primitives)
def _g_rot90(g): return [list(r) for r in zip(*g[::-1])]
def _g_flip(g): return g[::-1]
def _g_inv(g): return [[1-c if c in (0,1) else c for c in r] for r in g]
def _g_get(g, r, c): return g[r%len(g)][c%len(g[0])] if g and g[0] else 0

SAFE_FUNCS.update({
    'rot90': _g_rot90, 'flip': _g_flip, 'inv': _g_inv, 'get': _g_get
})
# Init Grammar weights for new ops
for k in ['rot90', 'flip', 'inv', 'get']:
    GRAMMAR_PROBS[k] = 1.0

class StepLimitExceeded(Exception): pass

class StepLimitTransformer(ast.NodeTransformer):
    """Injects step counting into loops and function calls to prevent halts."""
    def __init__(self, limit: int=1000):
        self.limit = limit
    
    def visit_FunctionDef(self, node):
        check = ast.parse(f"if _steps > {self.limit}: raise StepLimitExceeded()").body[0]
        inc = ast.parse("_steps += 1").body[0]
        glob = ast.Global(names=['_steps'])
        node.body.insert(0, inc)
        node.body.insert(1, check)
        node.body.insert(0, glob)
        self.generic_visit(node)
        return node
        
    def visit_While(self, node):
        check = ast.parse(f"if _steps > {self.limit}: raise StepLimitExceeded()").body[0]
        inc = ast.parse("_steps += 1").body[0]
        node.body.insert(0, inc)
        node.body.insert(1, check)
        self.generic_visit(node)
        return node

    def visit_For(self, node):
        check = ast.parse(f"if _steps > {self.limit}: raise StepLimitExceeded()").body[0]
        inc = ast.parse("_steps += 1").body[0]
        node.body.insert(0, inc)
        node.body.insert(1, check)
        self.generic_visit(node)
        return node

class CodeValidator(ast.NodeVisitor):
    """Allow full python subset: assign, flow control, but no unsafe imports."""
    ALLOWED = (ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return,
               ast.Assign, ast.AugAssign, ast.Name, ast.Constant, ast.Expr,
               ast.If, ast.While, ast.For, ast.Break, ast.Continue,
               ast.BinOp, ast.UnaryOp, ast.Compare, ast.Call, ast.List, ast.Subscript, 
               ast.Index, ast.Load, ast.Store, ast.IfExp,
               ast.operator, ast.boolop, ast.unaryop, ast.cmpop)
               
    def __init__(self):
        self.ok, self.err = True, None
        
    def visit(self, node):
        if not isinstance(node, self.ALLOWED):
            self.ok, self.err = False, f"Forbidden: {type(node).__name__}"
            return
        if isinstance(node, ast.Name):
            if node.id.startswith('__') or node.id in ('open', 'eval', 'exec', 'compile'):
                self.ok, self.err = False, f"Forbidden name: {node.id}"
                return
        super().generic_visit(node)

    def generic_visit(self, node):
        super().generic_visit(node)

def validate_code(code: str) -> Tuple[bool, str]:
    try:
        tree = ast.parse(code)
        v = CodeValidator()
        v.visit(tree)
        return (v.ok, v.err)
    except Exception as e:
        return (False, str(e))

def node_count(code: str) -> int:
    try:
        return sum((1 for _ in ast.walk(ast.parse(code))))
    except:
        return 999

def safe_exec(code: str, x: float, timeout_steps: int=1000) -> float:
    """Execute code with step limit. Code must define 'run(x)'."""
    try:
        tree = ast.parse(code)
        # Injection
        transformer = StepLimitTransformer(timeout_steps)
        tree = transformer.visit(tree)
        ast.fix_missing_locations(tree)
        
        # Scope
        env = {'_steps': 0, 'StepLimitExceeded': StepLimitExceeded, **SAFE_FUNCS, **SAFE_BUILTINS}
        exec(compile(tree, '<lgp>', 'exec'), env)
        
        if 'run' not in env:
            return float('nan')
            
        res = env['run'](x) # [MOD] Removed float() cast for List support
        return res
    except StepLimitExceeded:
        return float('nan')
    except Exception:
        return float('nan')

def apply_patch_safe(target_file: str, new_content: str) -> bool:
    """[NEW] Phase 5: Test-Driven Repair (TDR) - Atomic Patching"""
    import shutil
    bak = target_file + ".bak"
    try:
        print(f"[TDR] Applying atomic patch to {target_file}...")
        
        # Create backup if file exists
        if os.path.exists(target_file):
            shutil.copy(target_file, bak)
            
        with open(target_file, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        # 1. Syntax Check
        try:
             ast.parse(new_content)
        except SyntaxError as e:
             raise ValueError(f"Syntax Error: {e}")
             
        # 2. Import Check (Regression)
        # Only check if it's a python file
        if target_file.endswith('.py'):
            # Run basic import test
            cmd = f'python -c "import sys; sys.path.append(\'.\'); import {os.path.basename(target_file)[:-3]} as m; print(m.__name__)"'
            ret = os.system(cmd)
            if ret != 0:
                raise RuntimeError("Import Verification Failed")

        print("[TDR] Patch Verified. Commit.")
        return True
    except Exception as e:
        print(f"[TDR] Patch Validation Failed: {e}. Rolling back.")
        shutil.move(bak, target_file)
        return False

def safe_exec_engine(code: str, context: Dict[str, Any], timeout_steps: int=5000) -> Any:
    """Execute meta-engine code (selection/crossover) with safety limits."""
    try:
        tree = ast.parse(str(code)) # Ensure code is string
        # Injection for limits
        transformer = StepLimitTransformer(timeout_steps)
        tree = transformer.visit(tree)
        ast.fix_missing_locations(tree)
        
        # Scope: Allow context (pool, rng) + SAFE utils
        env = {
            '_steps': 0, 
            'StepLimitExceeded': StepLimitExceeded, 
            'random': random, # Engine needs random
            'math': math,
            'max': max, 'min': min, 'len': len, 'sum': sum, 'sorted': sorted, 'int': int, 'float': float, 'list': list,
            **context
        }
        
        exec(compile(tree, '<engine>', 'exec'), env)
        
        # Assumption: Engine code sets a variable 'result' or returns via a 'run' wrapper
        # For simplicity, we assume engine code defines 'run(ctx)' or similar, OR we just look for 'result'
        # Let's Standardize: Engine code should be a function body 'def run(...): ...'
        if 'run' in env:
            # We call run() with no args as context is already in env or passed if needed
            # But wait, selection needs arguments.
            # Better: context IS the globals. code defines 'run'. we call 'run(**context)'? 
            # Or just 'run()'.
            return env['run']() 
        return None
    except Exception as e:
        # print(f"Engine Error: {e}")
        return None

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
    if len(p1) < 2 or len(p2) < 2: return p1
    idx_a = rng.randint(0, len(p1))
    idx_b = rng.randint(0, len(p2))
    return p1[:idx_a] + p2[idx_b:]
"""

DEFAULT_MUTATION_CODE = """
def run():
    # Placeholder for mutation policy
    return 'default'
"""

@dataclass
class TaskSpec:
    name: str = 'poly2'
    x_min: float = -3.0
    x_max: float = 3.0
    n_train: int = 96
    n_hold: int = 96
    noise: float = 0.01
    stress_mult: float = 3.0
    target_code: Optional[str] = None  # [NEW] Phase 3: Dynamic Target

TARGET_FNS = {
    'sort': lambda x: sorted(x),
    'reverse': lambda x: list(reversed(x)),
    'max': lambda x: max(x) if x else 0,
    'filter': lambda x: [v for v in x if v > 0], # Simple filter: positive only
    # [NEW] Phase 4: ARC Tasks
    'arc_ident': lambda x: x,
    'arc_rot90': lambda x: [list(r) for r in zip(*x[::-1])],
    'arc_inv': lambda x: [[1-c if c in (0,1) else c for c in r] for r in x],
    'poly2': lambda x: 0.7 * x * x - 0.2 * x + 0.3, 
    'poly3': lambda x: 0.3 * x ** 3 - 0.5 * x + 0.1, 
    'sinmix': lambda x: math.sin(x) + 0.3 * math.cos(2 * x), 
    'absline': lambda x: abs(x) + 0.2 * x
}

# [NEW] Phase 5: Real Kaggle ARC Data (File Loader)
ARC_GYM_PATH = os.path.join(os.path.dirname(__file__), 'ARC_GYM')

def load_arc_task(task_id: str) -> Dict:
    """Load raw ARC JSON task."""
    fname = task_id
    if not fname.endswith('.json'): fname += '.json'
    path = os.path.join(ARC_GYM_PATH, fname)
    if not os.path.exists(path): return {}
    with open(path, 'r') as f:
        return json.load(f)

def get_arc_tasks() -> List[str]:
    """Scan ARC_GYM directory for available tasks."""
    if not os.path.exists(ARC_GYM_PATH): return []
    return [f[:-5] for f in os.listdir(ARC_GYM_PATH) if f.endswith('.json')]

@dataclass
class Batch:
    x_tr: List[Any] # Changed from float to Any to support Lists
    y_tr: List[Any]
    x_ho: List[Any]
    y_ho: List[Any]
    x_st: List[Any]
    y_st: List[Any]

def sample_batch(rng: random.Random, t: TaskSpec) -> Batch:
    # Phase 3: Dynamic Code Target
    if t.target_code:
        f = lambda x: safe_exec(t.target_code, x)
    elif t.name in ('sort', 'reverse', 'filter', 'max'):
        f = TARGET_FNS.get(t.name)
        if not f: f = lambda x: sorted(x)
    else:
        f = TARGET_FNS.get(t.name, lambda x: x)
        
    
    # [NEW] Phase 5: Real ARC Data Loader (JSON)
    json_data = load_arc_task(t.name.replace('arc_', '')) # Strip prefix if needed
    if json_data:
        # Flatten train/test. Kaggle JSON keys are 'input'/'output'
        pairs = json_data.get('train', []) + json_data.get('test', [])
        
        x_tr, y_tr = [], []
        # Repeat to fill batch
        while len(x_tr) < 20 and pairs:
             for p in pairs:
                 x_tr.append(p['input'])
                 y_tr.append(p['output'])
                 
        if not x_tr: return None # Handle empty file
        
        # Simple split
        return Batch(x_tr[:20], y_tr[:20], x_tr[:10], y_tr[:10], x_tr[:5], y_tr[:5])

    elif t.name in ('sort', 'reverse', 'filter', 'max'):
        # List Sorting Task
        def gen_lists(k, min_len, max_len):
            data = []
            for _ in range(k):
                a = max(1, int(min_len))
                b = max(a, int(max_len))
                l = rng.randint(a, b)
                data.append([rng.randint(-100, 100) for _ in range(l)])
            return data
            
        # Use x_min/x_max as Length Range
        x_tr = gen_lists(t.n_train, t.x_min, t.x_max)
        x_ho = gen_lists(t.n_hold, t.x_min + 2, t.x_max + 2) # Holdout slightly longer
        x_st = gen_lists(t.n_hold, t.x_max + 5, t.x_max + 10) # Stress much longer
        
        # Ground Truth
        y_tr = [f(x) for x in x_tr]
        y_ho = [f(x) for x in x_ho]
        y_st = [f(x) for x in x_st]
        return Batch(x_tr, y_tr, x_ho, y_ho, x_st, y_st)

    elif t.name.startswith('arc_'):
        # [NEW] Grid Generation for ARC
        def gen_grids(k, dim):
            data = []
            for _ in range(k):
                # 3x3 to 5x5 binary grids
                g = [[rng.randint(0, 1) for _ in range(dim)] for _ in range(dim)]
                data.append(g)
            return data
            
        dim = int(t.x_min) if t.x_min > 0 else 3
        x_tr = gen_grids(20, dim)
        x_ho = gen_grids(10, dim)
        x_st = gen_grids(10, dim+1)
        
        y_tr = [f(x) for x in x_tr]
        y_ho = [f(x) for x in x_ho]
        y_st = [f(x) for x in x_st]
        return Batch(x_tr, y_tr, x_ho, y_ho, x_st, y_st)


    else:
        # Standard Regression
        xs = lambda n, a, b: [a + (b - a) * rng.random() for _ in range(n)]
        ys = lambda xv, n: [f(x) + rng.gauss(0, n) if n > 0 else f(x) for x in xv]
        half = 0.5 * (t.x_max - t.x_min)
        mid = 0.5 * (t.x_min + t.x_max)
        x_tr, x_ho = (xs(t.n_train, t.x_min, t.x_max), xs(t.n_hold, t.x_min, t.x_max))
        x_st = xs(t.n_hold, mid - half * t.stress_mult, mid + half * t.stress_mult)
        return Batch(x_tr, ys(x_tr, t.noise), x_ho, ys(x_ho, t.noise), x_st, ys(x_st, t.noise * t.stress_mult))

@dataclass
class Genome:
    statements: List[str]
    gid: str = ''
    parents: List[str] = field(default_factory=list)
    op_tag: str = 'init'
    birth_ms: int = 0

    @property
    def code(self) -> str:
        body = "\n    ".join(self.statements) if self.statements else "return x"
        return f"def run(x):\n    # {self.gid}\n    _steps=0\n    v0=x\n    {body}"

    def __post_init__(self):
        if not self.gid:
            self.gid = sha256(''.join(self.statements) + str(time.time()))[:12]
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
    err: str = None
SCORE_W_HOLD = 0.48
SCORE_W_STRESS = 0.3
SCORE_W_TRAIN = 0.05

def calc_error(p: Any, t: Any) -> float:
    """Recursively calculate squared error between prediction and target."""
    if isinstance(t, (int, float)):
        if isinstance(p, (int, float)):
            return (p - t) ** 2
        return 1e6 # Type mismatch penalty
    elif isinstance(t, list):
        if not isinstance(p, list):
            return 1e6
        # List comparison
        if len(p) != len(t):
            # Penalty for length mismatch
            return 1000.0 * abs(len(p) - len(t))
        # Element-wise error
        return sum(calc_error(pv, tv) for pv, tv in zip(p, t))
    return 1e6 # Unknown type penalty

def calc_loss_sort(p: List[Any], t: List[Any]) -> float:
    """Advanced Fitness for Sorting: Inversions + Element Matching."""
    if not isinstance(p, list): return 1e6
    if len(p) != len(t): return 1000.0 * abs(len(p) - len(t))
    
    # 1. Element Mismatch Penalty (Gradient for content)
    # Use simple histogram match or sort-match
    p_sorted = sorted(p) if all(isinstance(x, (int, float)) for x in p) else p
    t_sorted = sorted(t)
    content_loss = sum((a-b)**2 for a, b in zip(p_sorted, t_sorted))
    
    if content_loss > 0.1:
        return 1000.0 + content_loss # Penalize content mismatch heavily first
        
    # 2. Inversion Count (Gradient for order)
    # Kendall Tau distance approximates swap distance
    inversions = 0
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            if p[i] > p[j]:
                inversions += 1
    return float(inversions)

def calc_heuristic_loss(p: Any, t: Any, task_name: str) -> float:
    """Specialized Fitness functions for Hard Benchmarks."""
    # 1. Base Checks
    if task_name == 'sort': return calc_loss_sort(p, t)
    
    if isinstance(t, list):
        if not isinstance(p, list): return 1e6
        if len(p) != len(t): return 500.0 * abs(len(p) - len(t))
        
        # 2. Reverse Task
        if task_name == 'reverse':
            # Check if elements match in reverse order
            # Actually, standard element-wise error against Target (already reversed) works fine
            # But maybe add 'set match' bonus?
            return sum(calc_error(pv, tv) for pv, tv in zip(p, t))
            
        # 3. Filter Task
        if task_name == 'filter':
            # Set match is crucial here, order matters less? No, order usually matters.
            # Use standard list error
            return sum(calc_error(pv, tv) for pv, tv in zip(p, t))
            
    # [NEW] Phase 4: ARC Grid Loss
    if task_name.startswith('arc_'):
        if not isinstance(p, list) or not p or not isinstance(p[0], list): return 1000.0
        # Shape penalty
        if len(p) != len(t) or len(p[0]) != len(t[0]): 
            return 500.0 + abs(len(p)-len(t)) + abs(len(p[0])-len(t[0]))
        # Pixel mismatch count
        err = 0
        for r in range(len(t)):
            for c in range(len(t[0])):
                if p[r][c] != t[r][c]: err += 1
        return float(err)

    # Default to generic recursive error
    return calc_error(p, t)

def mse_exec(code: str, xs: List[Any], ys: List[Any], task_name: str='') -> Tuple[bool, float, str]:
    ok, err = validate_code(code)
    if not ok:
        return (False, float('inf'), err)
    try:
        total_err = 0.0
        for x, y in zip(xs, ys):
            pred = safe_exec(code, x)
            if pred is None: return (False, float('inf'), "No return")
            
            if task_name in ('sort', 'reverse', 'max', 'filter') or task_name.startswith('arc_'):
                total_err += calc_heuristic_loss(pred, y, task_name)
            else:
                total_err += calc_error(pred, y)
        
        return (True, total_err / max(1, len(xs)), None)
    except Exception as e:
        return (False, float('inf'), str(e))

def evaluate(g: Genome, b: Batch, task_name: str, lam: float=0.0001) -> EvalResult:
    code = g.code
    ok1, tr, e1 = mse_exec(code, b.x_tr, b.y_tr, task_name)
    ok2, ho, e2 = mse_exec(code, b.x_ho, b.y_ho, task_name)
    ok3, st, e3 = mse_exec(code, b.x_st, b.y_st, task_name)
    ok = ok1 and ok2 and ok3 and all((math.isfinite(v) for v in [tr, ho, st]))
    nodes = node_count(code)
    score = SCORE_W_HOLD * ho + SCORE_W_STRESS * st + SCORE_W_TRAIN * tr + lam * nodes
    return EvalResult(ok, tr, ho, st, nodes, score, e1 or e2 or e3)

def _pick_node(rng: random.Random, body: ast.AST) -> ast.AST:
    nodes = list(ast.walk(body))
    return rng.choice(nodes[1:]) if len(nodes) > 1 else body

def _to_src(body: ast.AST) -> str:
    try:
        return ast.unparse(body)
    except:
        return 'x'

def _random_expr(rng: random.Random, depth: int=0) -> str:
    # Phase 4: EDA-Guided Probabilistic Generation
    if depth > 2:
        return rng.choice(['x', 'v0', str(rng.randint(0, 9))])
    
    # Sample structure type from learned weights
    options = ['binop', 'call', 'const', 'var']
    weights = [GRAMMAR_PROBS.get(k, 1.0) for k in options]
    mtype = rng.choices(options, weights=weights, k=1)[0]
    
    if mtype == 'binop':
        op = rng.choice(['+', '-', '*', '/', '**'])
        return f"({_random_expr(rng, depth+1)} {op} {_random_expr(rng, depth+1)})"
    elif mtype == 'call':
        # Weighted function selection
        funcs = list(SAFE_FUNCS.keys())
        f_weights = [GRAMMAR_PROBS.get(f, 0.5) for f in funcs]
        fname = rng.choices(funcs, weights=f_weights, k=1)[0]
        return f"{fname}({_random_expr(rng, depth+1)})"
    elif mtype == 'const':
        return f"{rng.uniform(-2, 2):.2f}"
    else: # var
        return rng.choice(['x', 'v0'])

def op_insert_assign(rng: random.Random, stmts: List[str]) -> List[str]:
    new_stmts = stmts[:]
    idx = rng.randint(0, len(new_stmts))
    var = f"v{rng.randint(0, 3)}"
    expr = _random_expr(rng)
    new_stmts.insert(idx, f"{var} = {expr}")
    return new_stmts

def op_insert_if(rng: random.Random, stmts: List[str]) -> List[str]:
    if not stmts: return stmts
    new_stmts = stmts[:]
    idx = rng.randint(0, len(new_stmts)-1)
    cond = f"v{rng.randint(0, 3)} < {rng.randint(0, 10)}"
    block = [f"    {s}" for s in new_stmts[idx:idx+2]]
    new_stmts[idx:idx+2] = [f"if {cond}:"] + block
    return new_stmts

def op_insert_while(rng: random.Random, stmts: List[str]) -> List[str]:
    if not stmts: return stmts
    new_stmts = stmts[:]
    idx = rng.randint(0, len(new_stmts)-1)
    cond = f"v{rng.randint(0, 3)} < {rng.randint(0, 10)}"
    block = [f"    {s}" for s in new_stmts[idx:idx+2]]
    new_stmts[idx:idx+2] = [f"while {cond}:"] + block
    return new_stmts

def op_delete_stmt(rng: random.Random, stmts: List[str]) -> List[str]:
    if not stmts: return stmts
    new_stmts = stmts[:]
    new_stmts.pop(rng.randint(0, len(new_stmts)-1))
    return new_stmts

def op_modify_line(rng: random.Random, stmts: List[str]) -> List[str]:
    if not stmts: return stmts
    new_stmts = stmts[:]
    idx = rng.randint(0, len(new_stmts)-1)
    if '=' in new_stmts[idx]:
        var = new_stmts[idx].split('=')[0].strip()
        new_stmts[idx] = f"{var} = {_random_expr(rng)}"
    return new_stmts

def op_shrink(rng: random.Random, expr: str) -> str:
    try:
        tree = ast.parse(expr, mode='eval').body
        prims = [ast.Name(id='x', ctx=ast.Load()), ast.Constant(0.0), ast.Constant(1.0)]
        return _to_src(rng.choice(prims))
    except:
        return expr

def op_grow(rng: random.Random, expr: str) -> str:
    try:
        tree = ast.parse(expr, mode='eval').body
        new_term = rng.choice([ast.BinOp(left=tree, op=ast.Add(), right=ast.Constant(rng.gauss(0, 1))), ast.BinOp(left=tree, op=ast.Mult(), right=ast.Name(id='x', ctx=ast.Load()))])
        new = _to_src(new_term)
        ok, _ = validate_expr(new)
        return new if ok else expr
    except:
        return expr

def op_graft_library(rng: random.Random, expr: str) -> str:
    """Graft a library function (Gamma, Erf, etc.) into the expression."""
    try:
        tools = list(SAFE_FUNCS.keys())
        tool = rng.choice(tools)
        tree = ast.parse(expr, mode='eval').body
        sub = _pick_node(rng, tree)
        call = ast.Call(func=ast.Name(id=tool, ctx=ast.Load()), args=[sub], keywords=[])
        new = _to_src(call)
        ok, _ = validate_expr(new)
        return new if ok else expr
    except:
        return expr
def op_tweak_const(rng: random.Random, stmts: List[str]) -> List[str]:
    """Micro-mutation: Tweak numerical constants."""
    if not stmts: return stmts
    new_stmts = stmts[:]
    idx = rng.randint(0, len(new_stmts)-1)
    
    def _tweak_node(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            val = node.value
            if isinstance(val, bool): return node
            # Gaussian pertubation
            new_val = val + rng.gauss(0, 0.1 * abs(val) + 0.01)
            # Occasional sign flip or zeroing
            if rng.random() < 0.05: new_val = -val
            if rng.random() < 0.05: new_val = 0
            return ast.Constant(value=new_val)
        return node

    class TweakTransformer(ast.NodeTransformer):
        def visit_Constant(self, node):
            return _tweak_node(node)

    try:
        tree = ast.parse(new_stmts[idx], mode='exec')
        new_tree = TweakTransformer().visit(tree)
        new_stmts[idx] = _to_src(new_tree)
    except:
        pass
    return new_stmts

def op_change_binary(rng: random.Random, stmts: List[str]) -> List[str]:
    """Swap binary operators (+, -, *, /)."""
    if not stmts: return stmts
    new_stmts = stmts[:]
    idx = rng.randint(0, len(new_stmts)-1)
    
    pops = [ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow]
    
    class OpTransformer(ast.NodeTransformer):
        def visit_BinOp(self, node):
            if rng.random() < 0.5:
                # Replace operator
                new_op = rng.choice(pops)()
                return ast.BinOp(left=node.left, op=new_op, right=node.right)
            return node

    try:
        tree = ast.parse(new_stmts[idx], mode='exec')
        new_tree = OpTransformer().visit(tree)
        new_stmts[idx] = _to_src(new_tree)
    except:
        pass
    return new_stmts

def op_list_manipulation(rng: random.Random, stmts: List[str]) -> List[str]:
    """Insert list operations (swaps, access)."""
    if not stmts: return stmts
    new_stmts = stmts[:]
    idx = rng.randint(0, len(new_stmts))
    
    ops = [
        f"v{rng.randint(0,3)} = x[{rng.randint(0,2)}]", # Read
        f"if len(x) > {rng.randint(1,5)}: v{rng.randint(0,3)} = x[0]", # Safe Read
        "v0, v1 = v1, v0", # Swap
        f"v{rng.randint(0,3)} = sorted(x)" # Cheat/Hint
    ]
    new_stmts.insert(idx, rng.choice(ops))
    return new_stmts

OPERATORS: Dict[str, Callable[[random.Random, List[str]], List[str]]] = {
    'insert_assign': op_insert_assign,
    'insert_if': op_insert_if,
    'insert_while': op_insert_while,
    'delete_stmt': op_delete_stmt, 
    'modify_line': op_modify_line,
    'tweak_const': op_tweak_const,
    'change_binary': op_change_binary,
    'list_manip': op_list_manipulation # [NEW] For Sorting
}
PRIMITIVE_OPS = list(OPERATORS.keys())
OPERATORS_LIB: Dict[str, Dict] = {}

def apply_synthesized_op(rng: random.Random, stmts: List[str], steps: List[str]) -> List[str]:
    """Apply a sequence of primitive operators to create a compound mutation."""
    result = stmts
    for step in steps:
        if step in OPERATORS:
            result = OPERATORS[step](rng, result)
    return result

def synthesize_new_operator(rng: random.Random) -> Tuple[str, Dict]:
    """Runtime synthesis: create a NEW operator by combining primitives."""
    n_steps = rng.randint(2, 4)
    steps = [rng.choice(list(OPERATORS.keys())) for _ in range(n_steps)]
    name = f"synth_{sha256(''.join(steps) + str(time.time()))[:8]}"
    return (name, {'steps': steps, 'score': 0.0})

class SurrogateModel:

    def __init__(self, k: int=5):
        self.k = k
        self.memory: List[Tuple[List[float], float]] = []

    def _extract_features(self, code: str) -> List[float]:
        """Extract lightweight features from code string."""
        return [
            len(code),
            code.count('\n'),
            code.count('if '),
            code.count('while '),
            code.count('='),
            code.count('return '),
            code.count('(')
        ]

    def train(self, history: List[Dict]):
        """Train model on history."""
        self.memory = []
        for h in history[-200:]:
            # Support both 'expr' (old) and 'code' (new) keys for transition
            src = h.get('code') or h.get('expr')
            if src and 'score' in h and isinstance(h['score'], (int, float)):
                feat = self._extract_features(src)
                self.memory.append((feat, float(h['score'])))

    def predict(self, expr: str) -> float:
        """Predict score using weighted KNN."""
        if not self.memory:
            return 0.0
        target = self._extract_features(expr)
        dists = []
        for feat, score in self.memory:
            d = sum(((f1 - f2) ** 2 for f1, f2 in zip(target, feat))) ** 0.5
            dists.append((d, score))
        dists.sort(key=lambda x: x[0])
        nearest = dists[:self.k]
        total_w = 0.0
        weighted_score = 0.0
        for d, s in nearest:
            w = 1.0 / (d + 1e-06)
            weighted_score += s * w
            total_w += w
        return weighted_score / total_w if total_w > 0 else 0.0
SURROGATE = SurrogateModel()

def update_operators_lib(op_name: str, delta_score: float):
    """Update operator library scores based on performance."""
    if op_name in OPERATORS_LIB:
        OPERATORS_LIB[op_name]['score'] += delta_score

class MAPElitesArchive:

    def __init__(self):
        self.grid: Dict[Tuple[int, int], Tuple[float, Genome]] = {}

    def _features(self, code: str) -> Tuple[int, int]:
        """Map code to feature bin coordinates."""
        l = len(code)
        l_bin = min(20, l // 20) # coarser binning
        d = code.count('\n')
        d_bin = min(10, d // 2)
        return (l_bin, d_bin)

    def add(self, genome: Genome, score: float):
        """Add genome to grid if it's the best in its cell."""
        feat = self._features(genome.code)
        if feat not in self.grid or score < self.grid[feat][0]:
            self.grid[feat] = (score, genome)

    def sample(self, rng: random.Random) -> Optional[Genome]:
        """Return a random elite from the grid."""
        if not self.grid:
            return None
        return rng.choice(list(self.grid.values()))[1]

    def snapshot(self) -> Dict:
        return {'grid_size': len(self.grid), 'entries': [(list(k), v[0], asdict(v[1])) for k, v in self.grid.items()]}

    @staticmethod
    def from_snapshot(s: Dict) -> 'MAPElitesArchive':
        ma = MAPElitesArchive()
        for k, score, g_dict in s.get('entries', []):
            ma.grid[tuple(k)] = (score, Genome(**g_dict))
        return ma
MAP_ELITES = MAPElitesArchive()

def save_map_elites(path: Path):
    """Save MAP-Elites grid to JSON."""
    path.write_text(json.dumps(MAP_ELITES.snapshot(), indent=2), encoding='utf-8')

def load_map_elites(path: Path):
    """Load MAP-Elites grid from JSON."""
    if path.exists():
        global MAP_ELITES
        try:
            MAP_ELITES = MAPElitesArchive.from_snapshot(json.loads(path.read_text(encoding='utf-8')))
        except:
            pass

def evolve_operator_meta(rng: random.Random) -> Tuple[str, Dict]:
    """Evolve a new operator by recombining existing high-performing ones (Meta-GP)."""
    candidates = [v for k, v in OPERATORS_LIB.items() if v.get('score', 0) > -5.0]
    if len(candidates) < 2:
        return synthesize_new_operator(rng)
    p1 = rng.choice(candidates)['steps']
    p2 = rng.choice(candidates)['steps']
    cut = rng.randint(0, min(len(p1), len(p2)))
    child_steps = p1[:cut] + p2[cut:]
    if rng.random() < 0.5:
        mut_type = rng.choice(['mod', 'add', 'del'])
        if mut_type == 'mod' and child_steps:
            child_steps[rng.randint(0, len(child_steps) - 1)] = rng.choice(PRIMITIVE_OPS)
        elif mut_type == 'add':
            child_steps.insert(rng.randint(0, len(child_steps)), rng.choice(PRIMITIVE_OPS))
        elif mut_type == 'del' and len(child_steps) > 1:
            child_steps.pop(rng.randint(0, len(child_steps) - 1))
    child_steps = child_steps[:6]
    if not child_steps:
        child_steps = [rng.choice(PRIMITIVE_OPS)]
    name = f"evo_{sha256(''.join(child_steps) + str(time.time()))[:8]}"
    return (name, {'steps': child_steps, 'score': 0.0})

def maybe_evolve_operators_lib(rng: random.Random, threshold: int=10):
    """Evolve the operator library: remove worst, add evolved ones."""
    sorted_ops = sorted(OPERATORS_LIB.items(), key=lambda x: x[1].get('score', 0))
    if len(OPERATORS_LIB) > 3:
        worst_name, worst_spec = sorted_ops[0]
        if worst_spec.get('score', 0) < -threshold:
            del OPERATORS_LIB[worst_name]
    if len(OPERATORS_LIB) < 8:
        if rng.random() < 0.7 and len(OPERATORS_LIB) >= 2:
            name, spec = evolve_operator_meta(rng)
        else:
            name, spec = synthesize_new_operator(rng)
        OPERATORS_LIB[name] = spec
        return name
        OPERATORS_LIB[name] = spec
        return name
    return None

class ProblemGenerator:
    """Phase 3: Co-evolutionary Discriminator. Generates hard tasks via Parameter Curriculum."""
    def __init__(self):
        self.archive: List[Dict] = []
    
    def evolve_task(self, rng: random.Random, current_elites: List[Genome]) -> TaskSpec:
        """Phase 3: Curriculum Learning (Parameter Evolution)."""
        # Guaranteed Solvability: Simply increase difficulty of known solvable tasks
        
        
        # 1. Curriculum Selection
        arc_tasks = get_arc_tasks()  # Get available JSON files
        base_options = ['sort', 'reverse', 'max', 'filter']
        
        # Prepend 'arc_' to JSON task IDs for consistency
        arc_options = [f'arc_{tid}' for tid in arc_tasks] if arc_tasks else []
        
        options = base_options + arc_options
        base_name = rng.choice(options) if options else 'sort'
        
        # 2. Difficulty Parameters
        level = rng.randint(1, 3)
        mn = 3 + level
        mx = 5 + level
        
        # For ARC, dimensions are handled by JSON data itself
        if base_name.startswith('arc_'):
             mn, mx = 3, 5  # Placeholder dims
             
        new_task = TaskSpec(
            name=base_name,
            n_train=64,
            n_hold=32,
            x_min=float(mn),
            x_max=float(mx),
            noise=0.0
        )
        return new_task
                
    def _mutate_target(self, rng: random.Random, code: str) -> str:
        return "deprecated"
        # 3. Create TaskSpec
        name = f"gen_task_{sha256(code)[:6]}"
        task = TaskSpec(name=name, target_code=code, x_min=-5.0, x_max=5.0)
        return task

def _rewrite_operators_block(src: str, new_lib: Dict) -> str:
    """Rewrite OPERATORS_LIB in source code with learned operators."""
    pattern = '(# @@OPERATORS_LIB_START@@\\s*\\nOPERATORS_LIB:\\s*Dict\\[str,\\s*Dict\\]\\s*=\\s*)(\\{[^}]*\\})(\\s*\\n# @@OPERATORS_LIB_END@@)'
    match = re.search(pattern, src, flags=re.DOTALL)
    if not match:
        return src
    prefix, _, suffix = (match.group(1), match.group(2), match.group(3))
    lines = ['{']
    for name, spec in new_lib.items():
        lines.append(f'    "{name}": {json.dumps(spec)},')
    lines.append('}')
    new_dict = '\n'.join(lines)
    return src[:match.start()] + prefix + new_dict + suffix + src[match.end():]

def save_operators_lib(path: Path):
    """Save OPERATORS_LIB to JSON file."""
    path.write_text(json.dumps(OPERATORS_LIB, indent=2), encoding='utf-8')

def load_operators_lib(path: Path):
    """Load OPERATORS_LIB from JSON file."""
    global OPERATORS_LIB
    if path.exists():
        try:
            OPERATORS_LIB.update(json.loads(path.read_text(encoding='utf-8')))
        except:
            pass

def crossover(rng: random.Random, a: List[str], b: List[str]) -> List[str]:
    if len(a) < 2 or len(b) < 2: return a
    idx_a = rng.randint(0, len(a))
    idx_b = rng.randint(0, len(b))
    return a[:idx_a] + b[idx_b:]

def seed_genome(rng: random.Random) -> Genome:
    # simple start
    return Genome(statements=["return x"])

@dataclass
class LearnedFunc:
    name: str
    expr: str
    trust: float = 1.0
    uses: int = 0

class FunctionLibrary:

    def __init__(self, max_size: int=16):
        self.funcs: Dict[str, LearnedFunc] = {}
        self.max_size = max_size

    def maybe_adopt(self, rng: random.Random, expr: str, threshold: float=0.1) -> Optional[str]:
        if len(self.funcs) >= self.max_size or rng.random() > threshold:
            return None
        try:
            tree = ast.parse(expr, mode='eval').body
            nodes = list(ast.walk(tree))
            if len(nodes) < 4:
                return None
            sub = _pick_node(rng, tree)
            sub_expr = _to_src(sub)
            if node_count(sub_expr) < 3:
                return None
            ok, _ = validate_expr(sub_expr)
            if not ok:
                return None
            name = f'h{len(self.funcs) + 1}'
            self.funcs[name] = LearnedFunc(name=name, expr=sub_expr)
            return name
        except:
            return None

    def maybe_inject(self, rng: random.Random, expr: str) -> Tuple[str, Optional[str]]:
        if not self.funcs or rng.random() > 0.2:
            return (expr, None)
        fn = rng.choice(list(self.funcs.values()))
        fn.uses += 1
        try:
            call = f'{fn.name}(x)'
            new = expr.replace('x', call, 1) if rng.random() < 0.5 else f'({expr}+{call})'
            ok, _ = validate_expr(new, extra=set(self.funcs.keys()))
            return (new, fn.name) if ok else (expr, None)
        except:
            return (expr, None)

    def update_trust(self, name: str, improved: bool):
        if name in self.funcs:
            self.funcs[name].trust *= 1.1 if improved else 0.9
            self.funcs[name].trust = clamp(self.funcs[name].trust, 0.1, 10.0)

    def get_helpers(self) -> Dict[str, Callable]:
        return {n: (lambda e: lambda x: safe_eval(e, x))(f.expr) for n, f in self.funcs.items()}

    def snapshot(self) -> Dict:
        return {'funcs': [asdict(f) for f in self.funcs.values()]}

    @staticmethod
    def from_snapshot(s: Dict) -> 'FunctionLibrary':
        lib = FunctionLibrary()
        for fd in s.get('funcs', []):
            lib.funcs[fd['name']] = LearnedFunc(**fd)
        return lib
        return lib

def induce_grammar(pool: List[Genome]):
    """Phase 4: Analyze top genomes to update GRAMMAR_PROBS (EDA step)."""
    # 1. Select Elites (Top 20%)
    if not pool: return
    elites = pool[:max(10, len(pool)//5)]
    
    # 2. Reset Counts
    counts = {k: 0.1 for k in GRAMMAR_PROBS} # Decay old beliefs slightly, keep priors
    
    # 3. Walk ASTs
    for g in elites:
        try:
            tree = ast.parse(g.code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id in counts:
                        counts[node.func.id] += 1.0
                    counts['call'] += 1.0
                elif isinstance(node, ast.BinOp):
                    counts['binop'] += 1.0
                elif isinstance(node, ast.Name) and node.id == 'x':
                    counts['var'] += 1.0
                elif isinstance(node, ast.Constant):
                    counts['const'] += 1.0
        except:
            pass
            
    # 4. Normalize & Update Global
    total = sum(counts.values())
    if total > 0:
        for k in counts:
            # Learning Rate 0.2 (Moving Average)
            old = GRAMMAR_PROBS.get(k, 1.0)
            target = (counts[k] / total) * 100.0 # Scale up for readability
            GRAMMAR_PROBS[k] = 0.8 * old + 0.2 * target

@dataclass
class MetaState:
    op_weights: Dict[str, float] = field(default_factory=lambda: {k: 1.0 for k in OPERATORS})
    mutation_rate: float = 0.8863
    crossover_rate: float = 0.2141
    complexity_lambda: float = 0.0001
    epsilon_explore: float = 0.15
    stuck_counter: int = 0
    strategy: EngineStrategy = field(default_factory=lambda: EngineStrategy(
        selection_code=DEFAULT_SELECTION_CODE, 
        crossover_code=DEFAULT_CROSSOVER_CODE, 
        mutation_policy_code=DEFAULT_MUTATION_CODE
    ))

    def sample_op(self, rng: random.Random) -> str:
        if rng.random() < self.epsilon_explore:
            return rng.choice(list(OPERATORS.keys()))
        total = sum((max(0.01, w) for w in self.op_weights.values()))
        r = rng.random() * total
        acc = 0.0
        for k, w in self.op_weights.items():
            acc += max(0.01, w)
            if r <= acc:
                return k
        return rng.choice(list(OPERATORS.keys()))

    def update(self, op: str, delta: float, accepted: bool):
        if op in self.op_weights:
            reward = max(0, -delta) if accepted else -0.1
            self.op_weights[op] = clamp(self.op_weights[op] + 0.1 * reward, 0.1, 5.0)
        if not accepted:
            self.stuck_counter += 1
            if self.stuck_counter > 20:
                self.epsilon_explore = clamp(self.epsilon_explore + 0.02, 0.1, 0.4)
                self.mutation_rate = clamp(self.mutation_rate + 0.03, 0.4, 0.95)
        else:
            self.stuck_counter = 0
            self.epsilon_explore = clamp(self.epsilon_explore - 0.01, 0.05, 0.3)


def induce_grammar(pool: List[Genome]):
    """Phase 4: Analyze top genomes to update GRAMMAR_PROBS (EDA step)."""
    # 1. Select Elites (Top 20%)
    if not pool: return
    elites = pool[:max(10, len(pool)//5)]
    
    # 2. Reset Counts
    counts = {k: 0.1 for k in GRAMMAR_PROBS}
    
    # 3. Walk ASTs
    for g in elites:
        try:
            tree = ast.parse(g.code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id in counts:
                        counts[node.func.id] += 1.0
                    counts['call'] += 1.0
                elif isinstance(node, ast.BinOp):
                    counts['binop'] += 1.0
                elif isinstance(node, ast.Name) and node.id == 'x':
                    counts['var'] += 1.0
                elif isinstance(node, ast.Constant):
                    counts['const'] += 1.0
        except:
            pass
            
    # 4. Update Global
    total = sum(counts.values())
    if total > 0:
        for k in counts:
            old = GRAMMAR_PROBS.get(k, 1.0)
            target = (counts[k] / total) * 100.0
            GRAMMAR_PROBS[k] = 0.8 * old + 0.2 * target

@dataclass
class Universe:
    uid: int
    seed: int
    meta: MetaState
    pool: List[Genome]
    library: FunctionLibrary
    discriminator: ProblemGenerator = field(default_factory=ProblemGenerator) # [NEW] Phase 3
    best: Optional[Genome] = None
    best_score: float = float('inf')
    best_hold: float = float('inf')
    best_stress: float = float('inf')
    history: List[Dict] = field(default_factory=list)

    def step(self, gen: int, task: TaskSpec, pop_size: int) -> Dict:
        rng = random.Random(self.seed + gen * 1009)
        batch = sample_batch(rng, task)
        # helpers = self.library.get_helpers() # Disable LGP library for now
        # Evaluation
        scored = []
        for g in self.pool:
            res = evaluate(g, batch, task.name, self.meta.complexity_lambda)
            if res.ok:
                scored.append((g, res))
        if not scored:
            self.pool = [seed_genome(rng) for _ in range(pop_size)]
            return {'gen': gen, 'accepted': False, 'reason': 'reseed'}
        scored.sort(key=lambda t: t[1].score)
        if scored:
            best_g, best_res = scored[0]
            MAP_ELITES.add(best_g, best_res.score)
        
        # Dynamic Selection
        sel_ctx = {
            'pool': [g for g, _ in scored],
            'scores': [res.score for _, res in scored],
            'pop_size': pop_size,
            'map_elites': MAP_ELITES,
            'rng': rng
        }
        sel_res = safe_exec_engine(self.meta.strategy.selection_code, sel_ctx)
        
        if sel_res and isinstance(sel_res, (tuple, list)) and len(sel_res) == 2:
            elites, parenting_pool = sel_res
        else:
            # Fallback
            elites = [g for g, _ in scored[:max(4, pop_size // 10)]]
            parenting_pool = [rng.choice(elites) for _ in range(pop_size - len(elites))]

        # [PHASE 3] Intelligent Guidance: Generate & Filter
        candidates: List[Genome] = []
        needed = pop_size - len(elites)
        # Generate 2x candidates to allow surrogate to pick best
        attempts_needed = needed * 2 
        
        mate_pool = list(elites) + list(parenting_pool)
        
        while len(candidates) < attempts_needed:
            # Pick Parent
            parent = rng.choice(parenting_pool) if parenting_pool else rng.choice(elites)
            
            new_stmts = None
            op_tag = 'copy'
            
            # Dynamic Crossover
            if rng.random() < self.meta.crossover_rate and len(mate_pool) > 1:
                p2 = rng.choice(mate_pool)
                cross_ctx = {'p1': parent.statements, 'p2': p2.statements, 'rng': rng}
                new_stmts = safe_exec_engine(self.meta.strategy.crossover_code, cross_ctx)
                if new_stmts:
                    op_tag = 'crossover'

            if not new_stmts:
                new_stmts = parent.statements[:]
            
            # Mutation
            if op_tag == 'copy' and rng.random() < self.meta.mutation_rate:
                use_synth = rng.random() < 0.3 and OPERATORS_LIB
                if use_synth:
                    synth_name = rng.choice(list(OPERATORS_LIB.keys()))
                    steps = OPERATORS_LIB[synth_name].get('steps', [])
                    new_stmts = apply_synthesized_op(rng, new_stmts, steps)
                    op_tag = f'synth:{synth_name}'
                else:
                    op = self.meta.sample_op(rng)
                    if op in OPERATORS:
                        new_stmts = OPERATORS[op](rng, new_stmts)
                    op_tag = f'mut:{op}'

            candidates.append(Genome(statements=new_stmts, parents=[parent.gid], op_tag=op_tag))

        # Surrogate Selection
        # Predict fitness for all candidates
        with_pred = []
        for c in candidates:
            score_est = SURROGATE.predict(c.code)
            with_pred.append((c, score_est))
            
        # Select top 'needed' by predicted score (ascending error)
        with_pred.sort(key=lambda x: x[1])
        selected_children = [c for c, _ in with_pred[:needed]]
        
        self.pool = list(elites) + selected_children
        if rng.random() < 0.02:
            maybe_evolve_operators_lib(rng)
            
        # [PHASE 4] EDA
        if gen % 5 == 0:
            induce_grammar(list(elites))

        best_g, best_res = scored[0]
        old_score = self.best_score
        accepted = best_res.score < self.best_score - 1e-09
        if accepted:
            self.best = best_g
            self.best_score = best_res.score
            self.best_hold = best_res.hold
            self.best_stress = best_res.stress
        op_used = best_g.op_tag.split(':')[1].split('|')[0] if ':' in best_g.op_tag else 'unknown'
        self.meta.update(op_used, self.best_score - old_score, accepted)
        log = {'gen': gen, 'accepted': accepted, 'score': self.best_score, 'hold': self.best_hold, 'stress': self.best_stress, 'code': self.best.code if self.best else 'none'}
        self.history.append(log)
        if gen % 5 == 0:
            SURROGATE.train(self.history)
        return log

    def co_evolve_step(self, gen: int, current_task: TaskSpec, pop_size: int) -> Tuple[Dict, Optional[TaskSpec]]:
        """Phase 3 Loop: Identify if we need a new task."""
        # 1. Run standard solver step
        log = self.step(gen, current_task, pop_size)
        
        # 2. Check if current task is "Solved" (Error < 1.0)
        # If solved, let Discriminator generate a harder task
        new_task = None
        if self.best_score < 1.0:
            # Task Solved! Challenge me.
            rng = random.Random(self.seed + gen)
            new_task = self.discriminator.evolve_task(rng, self.pool[:5])
            # Reset score to force adaptation to new task
            self.best_score = float('inf')
            self.pool = [seed_genome(rng) for _ in range(pop_size)] # Reseed for fairness
            
        return log, new_task

    def snapshot(self) -> Dict:
        return {'uid': self.uid, 'seed': self.seed, 'meta': asdict(self.meta), 'best': asdict(self.best) if self.best else None, 'best_score': self.best_score, 'best_hold': self.best_hold, 'best_stress': self.best_stress, 'pool': [asdict(g) for g in self.pool[:20]], 'library': self.library.snapshot(), 'history': self.history[-50:]}

    @staticmethod
    def from_snapshot(s: Dict) -> 'Universe':
        meta = MetaState(**{k: v for k, v in s.get('meta', {}).items() if k != 'op_weights'})
        meta.op_weights = s.get('meta', {}).get('op_weights', {k: 1.0 for k in OPERATORS})
        pool = [Genome(**g) for g in s.get('pool', [])]
        lib = FunctionLibrary.from_snapshot(s.get('library', {}))
        u = Universe(uid=s.get('uid', 0), seed=s.get('seed', 0), meta=meta, pool=pool, library=lib)
        if s.get('best'):
            u.best = Genome(**s['best'])
        u.best_score = s.get('best_score', float('inf'))
        u.best_hold = s.get('best_hold', float('inf'))
        u.best_stress = s.get('best_stress', float('inf'))
        u.history = s.get('history', [])
        return u

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
STATE_DIR = Path('.rsi_state')

def save_state(gs: GlobalState):
    gs.updated_ms = now_ms()
    write_json(STATE_DIR / 'state.json', asdict(gs))
    save_operators_lib(STATE_DIR / 'operators_lib.json')
    save_map_elites(STATE_DIR / 'map_elites.json')

def load_state() -> Optional[GlobalState]:
    p = STATE_DIR / 'state.json'
    if not p.exists():
        return None
    try:
        load_operators_lib(STATE_DIR / 'operators_lib.json')
        load_map_elites(STATE_DIR / 'map_elites.json')
        return GlobalState(**read_json(p))
    except:
        return None

def run_multiverse(seed: int, task: TaskSpec, gens: int, pop: int, n_univ: int, resume: bool=False, save_every: int=5) -> GlobalState:
    safe_mkdir(STATE_DIR)
    if resume and (gs0 := load_state()):
        us = [Universe.from_snapshot(s) for s in gs0.universes]
        start = gs0.generations_done
    else:
        us = [Universe(uid=i, seed=seed + i * 9973, meta=MetaState(), pool=[seed_genome(random.Random(seed + i)) for _ in range(pop)], library=FunctionLibrary()) for i in range(n_univ)]
        start = 0
    for gen in range(start, start + gens):
        for u in us:
            u.step(gen, task, pop)
        us.sort(key=lambda u: u.best_score)
        best = us[0]
        print(f"[Gen {gen + 1:4d}] Score: {best.best_score:.4f} | Hold: {best.best_hold:.4f} | Stress: {best.best_stress:.4f} | {(best.best.code if best.best else 'none')}")
        if save_every > 0 and (gen + 1) % save_every == 0:
            gs = GlobalState('RSI_EXT_v1', now_ms(), now_ms(), seed, asdict(task), [u.snapshot() for u in us], us[0].uid, gen + 1)
            save_state(gs)
    gs = GlobalState('RSI_EXT_v1', now_ms(), now_ms(), seed, asdict(task), [u.snapshot() for u in us], us[0].uid, start + gens)
    save_state(gs)
    return gs
PATCH_LEVELS = {0: 'hyperparameter', 1: 'op_weight', 2: 'operator_toggle', 3: 'eval_weight', 4: 'operator_inject', 5: 'meta_logic'}

@dataclass
class PatchPlan:
    level: int
    patch_id: str
    title: str
    rationale: str
    new_source: str
    diff: str

def _read_self() -> str:
    return Path(__file__).read_text(encoding='utf-8')

def _patch_dataclass(src: str, cls: str, field: str, val: Any) -> Tuple[bool, str]:
    try:
        mod = ast.parse(src)
        patched = False
        for node in ast.walk(mod):
            if isinstance(node, ast.ClassDef) and node.name == cls:
                for stmt in node.body:
                    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and (stmt.target.id == field):
                        stmt.value = ast.Constant(value=val)
                        patched = True
        if not patched:
            return (False, src)
        ast.fix_missing_locations(mod)
        return (True, ast.unparse(mod))
    except:
        return (False, src)

def _patch_global_const(src: str, name: str, val: float) -> Tuple[bool, str]:
    pattern = f'^({name}\\s*=\\s*)[\\d.eE+-]+'
    new_src, n = re.subn(pattern, f'\\g<1>{val}', src, flags=re.MULTILINE)
    return (n > 0, new_src)

def _patch_add_operator(src: str, op_name: str, op_code: str) -> Tuple[bool, str]:
    fn_code = f'\ndef {op_name}(rng: random.Random, expr: str) -> str:\n    {op_code}\n'
    reg_line = f'    "{op_name[3:]}": {op_name},'
    if 'OPERATORS:' in src:
        insert_pos = src.find('OPERATORS:')
        bracket_pos = src.find('{', insert_pos)
        new_src = src[:bracket_pos + 1] + '\n' + reg_line + src[bracket_pos + 1:]
        fn_insert = src.rfind('\n', 0, insert_pos)
        new_src = new_src[:fn_insert] + fn_code + new_src[fn_insert:]
        try:
            ast.parse(new_src)
            return (True, new_src)
        except:
            pass
    return (False, src)

def propose_patches(gs: GlobalState, levels: List[int]) -> List[PatchPlan]:
    src = _read_self()
    plans: List[PatchPlan] = []
    rng = random.Random(gs.updated_ms)
    best_u = next((s for s in gs.universes if s.get('uid') == gs.selected_uid), gs.universes[0] if gs.universes else {})
    meta = best_u.get('meta', {})
    if 0 in levels:
        for cls, field, base in [('MetaState', 'mutation_rate', 0.65), ('MetaState', 'crossover_rate', 0.2), ('MetaState', 'epsilon_explore', 0.15)]:
            cur = meta.get(field, base)
            new_val = round(clamp(cur * rng.uniform(0.85, 1.15), 0.1, 0.95), 4)
            ok, new_src = _patch_dataclass(src, cls, field, new_val)
            if ok and new_src != src:
                plans.append(PatchPlan(0, sha256(f'{cls}.{field}={new_val}')[:8], f'L0: {cls}.{field} -> {new_val}', 'Adaptive tuning', new_src, unified_diff(src, new_src, 'script.py')))
    if 1 in levels:
        op_w = meta.get('op_weights', {})
        if op_w:
            for op, w in list(op_w.items())[:3]:
                new_w = round(clamp(w * rng.uniform(0.9, 1.1), 0.1, 5.0), 3)
                op_w[op] = new_w
            pass
    if 3 in levels:
        for name, base in [('SCORE_W_HOLD', 0.6), ('SCORE_W_STRESS', 0.35), ('SCORE_W_TRAIN', 0.05)]:
            new_val = round(clamp(base * rng.uniform(0.8, 1.2), 0.05, 0.9), 2)
            ok, new_src = _patch_global_const(src, name, new_val)
            if ok and new_src != src:
                plans.append(PatchPlan(3, sha256(f'{name}={new_val}')[:8], f'L3: {name} -> {new_val}', 'Eval rebalancing', new_src, unified_diff(src, new_src, 'script.py')))
    if 4 in levels:
        if OPERATORS_LIB:
            new_src = _rewrite_operators_block(src, OPERATORS_LIB)
            if new_src != src:
                plans.append(PatchPlan(4, sha256(str(OPERATORS_LIB))[:8], f'L4: Persist {len(OPERATORS_LIB)} learned operators', f'Operators: {list(OPERATORS_LIB.keys())[:5]}', new_src, unified_diff(src, new_src, 'script.py')))
    if 5 in levels:
        elite_mods = [('max(4, pop_size // 10)', 'max(4, pop_size // 8)'), ('max(4, pop_size // 8)', 'max(4, pop_size // 12)'), ('max(4, pop_size // 12)', 'max(4, pop_size // 10)')]
        for old_pat, new_pat in elite_mods:
            if old_pat in src:
                new_src = src.replace(old_pat, new_pat, 1)
                plans.append(PatchPlan(5, sha256(new_pat)[:8], f'L5: elite_k ratio change', 'Meta: selection pressure', new_src, unified_diff(src, new_src, 'script.py')))
                break
        stuck_mods = [('self.stuck_counter > 20', 'self.stuck_counter > 15'), ('self.stuck_counter > 15', 'self.stuck_counter > 25'), ('self.stuck_counter > 25', 'self.stuck_counter > 20')]
        for old_pat, new_pat in stuck_mods:
            if old_pat in src:
                new_src = src.replace(old_pat, new_pat, 1)
                plans.append(PatchPlan(5, sha256(new_pat)[:8], f'L5: stuck threshold', 'Meta: stagnation detection', new_src, unified_diff(src, new_src, 'script.py')))
                break
        cand_mods = [('plans[:8]', 'plans[:10]'), ('plans[:10]', 'plans[:6]'), ('plans[:6]', 'plans[:8]')]
        for old_pat, new_pat in cand_mods:
            if old_pat in src:
                new_src = src.replace(old_pat, new_pat)
                plans.append(PatchPlan(5, sha256(new_pat)[:8], f'L5: candidate pool size', 'Meta: patch search breadth', new_src, unified_diff(src, new_src, 'script.py')))
                break
    rng.shuffle(plans)
    return plans[:8]

def probe_run(script: Path, gens: int=15, pop: int=32) -> float:
    with tempfile.TemporaryDirectory() as td:
        try:
            proc = subprocess.run([sys.executable, str(script), 'evolve', '--fresh', '--generations', str(gens), '--population', str(pop), '--universes', '1', '--state-dir', td], capture_output=True, text=True, timeout=90)
            for line in reversed(proc.stdout.splitlines()):
                if 'Score:' in line:
                    m = re.search('Score:\\s*([\\d.]+)', line)
                    if m:
                        return float(m.group(1))
        except:
            pass
    return float('inf')

def run_deep_autopatch(levels: List[int], candidates: int=4, apply: bool=False) -> Dict:
    gs = load_state()
    if not gs:
        return {'error': 'No state. Run evolve first.'}
    script = Path(__file__).resolve()
    baseline = probe_run(script)
    print(f'[AUTOPATCH L{levels}] Baseline: {baseline:.4f}')
    plans = propose_patches(gs, levels)[:candidates]
    if not plans:
        return {'error': 'No patches generated'}
    results = []
    best_plan, best_score = (None, baseline)
    for p in plans:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(p.new_source)
            tmp = Path(f.name)
        try:
            score = probe_run(tmp)
            improved = score < baseline - 1e-06
            results.append({'level': p.level, 'id': p.patch_id, 'title': p.title, 'score': score, 'improved': improved})
            print(f"[L{p.level}] {p.patch_id}: {p.title} -> {score:.4f} {('OK' if improved else 'FAIL')}")
            if improved and score < best_score:
                best_score, best_plan = (score, p)
        finally:
            tmp.unlink(missing_ok=True)
    if best_plan and apply:
        backup = script.with_suffix('.bak')
        if not backup.exists():
            backup.write_text(script.read_text(encoding='utf-8'), encoding='utf-8')
        script.write_text(best_plan.new_source, encoding='utf-8')
        print(f'[OK] Applied L{best_plan.level} patch: {best_plan.title}')
        return {'applied': best_plan.patch_id, 'level': best_plan.level, 'score': best_score, 'results': results}
    elif best_plan:
        out = STATE_DIR / 'patched.py'
        out.write_text(best_plan.new_source, encoding='utf-8')
        print(f'[OK] Best patch saved to {out}')
        return {'best': best_plan.patch_id, 'score': best_score, 'file': str(out), 'results': results}
    return {'improved': False, 'baseline': baseline, 'results': results}

def run_rsi_loop(gens_per_round: int, rounds: int, levels: List[int], pop: int, n_univ: int):
    """Continuous RSI: evolve -> autopatch -> evolve -> autopatch..."""
    task = TaskSpec()
    seed = int(time.time()) % 100000
    for r in range(rounds):
        print(f"\n{'=' * 60}\n[RSI ROUND {r + 1}/{rounds}]\n{'=' * 60}")
        print(f'[EVOLVE] {gens_per_round} generations...')
        run_multiverse(seed, task, gens_per_round, pop, n_univ, resume=r > 0)
        print(f'[AUTOPATCH] Trying L{levels}...')
        result = run_deep_autopatch(levels, candidates=4, apply=True)
        if result.get('applied'):
            print(f'[RSI] Self-modified! Reloading...')
    print(f'\n[RSI LOOP COMPLETE] {rounds} rounds finished')

def cmd_selftest(args):
    print('[selftest] Validating...')
    assert validate_expr('sin(x) + x*x')[0]
    assert not validate_expr("__import__('os')")[0]
    g = seed_genome(random.Random(42))
    b = sample_batch(random.Random(42), TaskSpec())
    r = evaluate(g, b)
    assert isinstance(r.score, float)
    print('[selftest] OK')
    return 0

def cmd_evolve(args):
    global STATE_DIR
    STATE_DIR = Path(args.state_dir)
    run_multiverse(args.seed, TaskSpec(name=args.task), args.generations, args.population, args.universes, args.resume and (not args.fresh), args.save_every)
    print(f"\n[OK] State saved to {STATE_DIR / 'state.json'}")
    return 0

def cmd_best(args):
    global STATE_DIR
    STATE_DIR = Path(args.state_dir)
    gs = load_state()
    if not gs:
        print('No state.')
        return 1
    u = next((s for s in gs.universes if s.get('uid') == gs.selected_uid), gs.universes[0] if gs.universes else {})
    print(f"Expr: {u.get('best', {}).get('expr', 'none')}")
    print(f"Score: {u.get('best_score')} | Hold: {u.get('best_hold')} | Stress: {u.get('best_stress')}")
    print(f'Generations: {gs.generations_done}')
    return 0

def cmd_autopatch(args):
    global STATE_DIR
    STATE_DIR = Path(args.state_dir)
    levels = [int(l) for l in args.levels.split(',')]
    result = run_deep_autopatch(levels, args.candidates, args.apply)
    print(json.dumps(result, indent=2, default=str))
    return 0

def cmd_rsi_loop(args):
    global STATE_DIR
    STATE_DIR = Path(args.state_dir)
    levels = [int(l) for l in args.levels.split(',')]
    run_rsi_loop(args.generations, args.rounds, levels, args.population, args.universes)
    return 0

def build_parser():
    p = argparse.ArgumentParser(prog='UNIFIED_RSI_EXTENDED', description='True RSI Engine with L0-L5 Self-Modification')
    sub = p.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('selftest')
    s.set_defaults(fn=cmd_selftest)
    e = sub.add_parser('evolve')
    e.add_argument('--seed', type=int, default=1337)
    e.add_argument('--generations', type=int, default=80)
    e.add_argument('--population', type=int, default=128)
    e.add_argument('--universes', type=int, default=4)
    e.add_argument('--task', default='poly2')
    e.add_argument('--resume', action='store_true')
    e.add_argument('--fresh', action='store_true')
    e.add_argument('--save-every', type=int, default=5)
    e.add_argument('--state-dir', default='.rsi_state')
    e.set_defaults(fn=cmd_evolve)
    b = sub.add_parser('best')
    b.add_argument('--state-dir', default='.rsi_state')
    b.set_defaults(fn=cmd_best)
    a = sub.add_parser('autopatch')
    a.add_argument('--levels', default='0,1,3')
    a.add_argument('--candidates', type=int, default=4)
    a.add_argument('--apply', action='store_true')
    a.add_argument('--state-dir', default='.rsi_state')
    a.set_defaults(fn=cmd_autopatch)
    r = sub.add_parser('rsi-loop')
    r.add_argument('--generations', type=int, default=50)
    r.add_argument('--rounds', type=int, default=5)
    r.add_argument('--levels', default='0,1,3')
    r.add_argument('--population', type=int, default=64)
    r.add_argument('--universes', type=int, default=2)
    r.add_argument('--state-dir', default='.rsi_state')
    r.set_defaults(fn=cmd_rsi_loop)
    return p

def main():
    return build_parser().parse_args().fn(build_parser().parse_args())
if __name__ == '__main__':
    raise SystemExit(main())