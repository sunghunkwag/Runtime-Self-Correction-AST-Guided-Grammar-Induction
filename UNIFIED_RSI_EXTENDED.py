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
    'sin': math.sin, 'cos': math.cos, 'tan': math.tan, 
    'exp': math.exp, 'tanh': math.tanh, 'abs': abs, 
    'sqrt': lambda x: math.sqrt(abs(x) + 1e-12), 
    'log': lambda x: math.log(abs(x) + 1e-12), 
    'pow2': lambda x: x * x, 
    'sigmoid': lambda x: 1.0 / (1.0 + math.exp(-clamp(x, -500, 500))),
    # Code Grafting Tools
    'gamma': lambda x: math.gamma(abs(x) + 1e-9) if abs(x) < 170 else float('inf'), # Limit to avoid overflow
    'erf': math.erf,
    'ceil': math.ceil, 'floor': math.floor,
    'sign': lambda x: math.copysign(1.0, x)
}
SAFE_BUILTINS = {'abs': abs, 'min': min, 'max': max, 'float': float, 'int': int}
SAFE_VARS = {'x'}

class ExprValidator(ast.NodeVisitor):
    ALLOWED = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name, ast.Call, ast.IfExp, ast.Compare, ast.Num)
    BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
    UNOPS = (ast.UAdd, ast.USub)
    CMPS = (ast.Lt, ast.Gt, ast.LtE, ast.GtE, ast.Eq)

    def __init__(self, extra: Set[str]=None):
        self.ok, self.err = (True, None)
        self.extra = extra or set()

    def fail(self, msg):
        self.ok, self.err = (False, msg)

    def generic_visit(self, node):
        if not isinstance(node, self.ALLOWED):
            self.fail(f'bad:{type(node).__name__}')
        else:
            super().generic_visit(node)

    def visit_Name(self, n):
        if n.id not in SAFE_VARS and n.id not in SAFE_FUNCS and (n.id not in self.extra):
            self.fail(f'name:{n.id}')

    def visit_BinOp(self, n):
        if not isinstance(n.op, self.BINOPS):
            self.fail(f'binop:{type(n.op).__name__}')
        else:
            self.visit(n.left)
            self.visit(n.right)

    def visit_UnaryOp(self, n):
        if not isinstance(n.op, self.UNOPS):
            self.fail(f'unop:{type(n.op).__name__}')
        else:
            self.visit(n.operand)

    def visit_Call(self, n):
        if not isinstance(n.func, ast.Name):
            self.fail('call')
        elif n.func.id not in SAFE_FUNCS and n.func.id not in self.extra:
            self.fail(f'fn:{n.func.id}')
        else:
            for a in n.args:
                self.visit(a)

def validate_expr(expr: str, extra: Set[str]=None) -> Tuple[bool, str]:
    try:
        tree = ast.parse(expr, mode='eval')
        v = ExprValidator(extra)
        v.visit(tree)
        return (True, None) if v.ok else (False, v.err)
    except Exception as e:
        return (False, str(e))

def node_count(expr: str) -> int:
    try:
        return sum((1 for _ in ast.walk(ast.parse(expr, mode='eval'))))
    except:
        return 999

def safe_eval(expr: str, x: float, helpers: Dict[str, Callable]=None) -> float:
    try:
        env = {'__builtins__': SAFE_BUILTINS, **SAFE_FUNCS, **(helpers or {}), 'x': float(x)}
        return float(eval(compile(ast.parse(expr, mode='eval'), '<e>', 'eval'), env))
    except:
        return float('nan')

@dataclass
class TaskSpec:
    name: str = 'poly2'
    x_min: float = -3.0
    x_max: float = 3.0
    n_train: int = 96
    n_hold: int = 96
    noise: float = 0.01
    stress_mult: float = 3.0
TARGET_FNS = {'poly2': lambda x: 0.7 * x * x - 0.2 * x + 0.3, 'poly3': lambda x: 0.3 * x ** 3 - 0.5 * x + 0.1, 'sinmix': lambda x: math.sin(x) + 0.3 * math.cos(2 * x), 'absline': lambda x: abs(x) + 0.2 * x}

@dataclass
class Batch:
    x_tr: List[float]
    y_tr: List[float]
    x_ho: List[float]
    y_ho: List[float]
    x_st: List[float]
    y_st: List[float]

def sample_batch(rng: random.Random, t: TaskSpec) -> Batch:
    f = TARGET_FNS.get(t.name, lambda x: x)
    xs = lambda n, a, b: [a + (b - a) * rng.random() for _ in range(n)]
    ys = lambda xv, n: [f(x) + rng.gauss(0, n) if n > 0 else f(x) for x in xv]
    half = 0.5 * (t.x_max - t.x_min)
    mid = 0.5 * (t.x_min + t.x_max)
    x_tr, x_ho = (xs(t.n_train, t.x_min, t.x_max), xs(t.n_hold, t.x_min, t.x_max))
    x_st = xs(t.n_hold, mid - half * t.stress_mult, mid + half * t.stress_mult)
    return Batch(x_tr, ys(x_tr, t.noise), x_ho, ys(x_ho, t.noise), x_st, ys(x_st, t.noise * t.stress_mult))

@dataclass
class Genome:
    expr: str
    gid: str = ''
    parents: List[str] = field(default_factory=list)
    op_tag: str = 'init'
    birth_ms: int = 0

    def __post_init__(self):
        if not self.gid:
            self.gid = sha256(self.expr + str(time.time()))[:12]
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

def mse(expr: str, xs: List[float], ys: List[float]) -> Tuple[bool, float, str]:
    ok, err = validate_expr(expr)
    if not ok:
        return (False, float('inf'), err)
    try:
        se = sum(((safe_eval(expr, x) - y) ** 2 for x, y in zip(xs, ys)))
        return (True, se / max(1, len(xs)), None)
    except Exception as e:
        return (False, float('inf'), str(e))

def evaluate(g: Genome, b: Batch, lam: float=0.0001) -> EvalResult:
    ok1, tr, e1 = mse(g.expr, b.x_tr, b.y_tr)
    ok2, ho, e2 = mse(g.expr, b.x_ho, b.y_ho)
    ok3, st, e3 = mse(g.expr, b.x_st, b.y_st)
    ok = ok1 and ok2 and ok3 and all((math.isfinite(v) for v in [tr, ho, st]))
    nodes = node_count(g.expr)
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

def op_const_drift(rng: random.Random, expr: str) -> str:
    try:
        tree = ast.parse(expr, mode='eval').body
        consts = [n for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, (int, float))]
        if consts:
            c = rng.choice(consts)
            c.value = float(c.value) + rng.gauss(0, 0.3)
        return _to_src(tree)
    except:
        return expr

def op_swap_binop(rng: random.Random, expr: str) -> str:
    try:
        tree = ast.parse(expr, mode='eval').body
        bins = [n for n in ast.walk(tree) if isinstance(n, ast.BinOp)]
        if bins:
            n = rng.choice(bins)
            n.op = rng.choice([ast.Add(), ast.Sub(), ast.Mult()])
        return _to_src(tree)
    except:
        return expr

def op_wrap_unary(rng: random.Random, expr: str) -> str:
    try:
        tree = ast.parse(expr, mode='eval').body
        sub = _pick_node(rng, tree)
        wrapped = ast.UnaryOp(op=rng.choice([ast.UAdd(), ast.USub()]), operand=sub)
        new = _to_src(wrapped)
        ok, _ = validate_expr(new)
        return new if ok else expr
    except:
        return expr

def op_wrap_call(rng: random.Random, expr: str) -> str:
    try:
        tree = ast.parse(expr, mode='eval').body
        sub = _pick_node(rng, tree)
        fn = rng.choice(list(SAFE_FUNCS.keys()))
        call = ast.Call(func=ast.Name(id=fn, ctx=ast.Load()), args=[sub], keywords=[])
        new = _to_src(call)
        ok, _ = validate_expr(new)
        return new if ok else expr
    except:
        return expr

def op_insert_ifexp(rng: random.Random, expr: str) -> str:
    try:
        tree = ast.parse(expr, mode='eval').body
        sub = _pick_node(rng, tree)
        thresh = ast.Constant(value=rng.uniform(-1, 1))
        test = ast.Compare(left=ast.Name(id='x', ctx=ast.Load()), ops=[ast.Lt()], comparators=[thresh])
        alt = ast.Constant(value=rng.uniform(-2, 2))
        ife = ast.IfExp(test=test, body=sub, orelse=alt)
        new = _to_src(ife)
        ok, _ = validate_expr(new)
        return new if ok else expr
    except:
        return expr

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
        tools = ['gamma', 'erf', 'ceil', 'floor', 'sign', 'sqrt', 'log']
        tool = rng.choice(tools)
        # Wrap a random node with the tool
        tree = ast.parse(expr, mode='eval').body
        sub = _pick_node(rng, tree)
        call = ast.Call(func=ast.Name(id=tool, ctx=ast.Load()), args=[sub], keywords=[])
        new = _to_src(call)
        ok, _ = validate_expr(new)
        return new if ok else expr
    except:
        return expr

OPERATORS: Dict[str, Callable[[random.Random, str], str]] = {
    'const_drift': op_const_drift, 'swap_binop': op_swap_binop, 
    'wrap_unary': op_wrap_unary, 'wrap_call': op_wrap_call, 
    'insert_ifexp': op_insert_ifexp, 'shrink': op_shrink, 'grow': op_grow,
    'graft_lib': op_graft_library
}
PRIMITIVE_OPS = ['const_drift', 'swap_binop', 'wrap_unary', 'wrap_call', 'shrink', 'grow', 'graft_lib']
OPERATORS_LIB: Dict[str, Dict] = {'synth_drift_then_grow': {'steps': ['const_drift', 'grow'], 'score': 0.0}, 'synth_shrink_wrap': {'steps': ['shrink', 'wrap_call'], 'score': 0.0}, 'synth_double_drift': {'steps': ['const_drift', 'const_drift'], 'score': 0.0}}

def apply_synthesized_op(rng: random.Random, expr: str, steps: List[str]) -> str:
    """Apply a sequence of primitive operators to create a compound mutation."""
    result = expr
    for step in steps:
        if step in OPERATORS:
            result = OPERATORS[step](rng, result)
    return result

def synthesize_new_operator(rng: random.Random) -> Tuple[str, Dict]:
    """Runtime synthesis: create a NEW operator by combining primitives."""
    n_steps = rng.randint(2, 4)
    steps = [rng.choice(PRIMITIVE_OPS) for _ in range(n_steps)]
    name = f"synth_{sha256(''.join(steps) + str(time.time()))[:8]}"
    return (name, {'steps': steps, 'score': 0.0})

# =============================================================================
# SURROGATE MODEL (RSI Intuition)
# Predicts expression performance to save evaluation costs
# =============================================================================

class SurrogateModel:
    def __init__(self, k: int = 5):
        self.k = k
        self.memory: List[Tuple[List[float], float]] = []  # (features, score)
        
    def _extract_features(self, expr: str) -> List[float]:
        """Extract lightweight features from expression string."""
        return [
            len(expr),
            expr.count('+'),
            expr.count('-'),
            expr.count('*'),
            expr.count('/'),
            expr.count('sin'),
            expr.count('cos'),
            expr.count('tanh'),
            expr.count('(')  # Rough proxy for depth/complexity
        ]
        
    def train(self, history: List[Dict]):
        """Train model on history (Just storing features for KNN)."""
        # Keep only recent valid entries
        self.memory = []
        for h in history[-200:]:  # Limit memory size
            if 'expr' in h and 'score' in h and isinstance(h['score'], (int, float)):
                 feat = self._extract_features(h['expr'])
                 self.memory.append((feat, float(h['score'])))
                 
    def predict(self, expr: str) -> float:
        """Predict score using weighted KNN."""
        if not self.memory:
            return 0.0
            
        target = self._extract_features(expr)
        
        # Calculate distances
        dists = []
        for feat, score in self.memory:
            # Euclidean distance approximation
            d = sum((f1 - f2) ** 2 for f1, f2 in zip(target, feat)) ** 0.5
            dists.append((d, score))
            
        # Get k nearest
        dists.sort(key=lambda x: x[0])
        nearest = dists[:self.k]
        
        # Weighted average
        total_w = 0.0
        weighted_score = 0.0
        for d, s in nearest:
            w = 1.0 / (d + 1e-6)
            weighted_score += s * w
            total_w += w
            
        return weighted_score / total_w if total_w > 0 else 0.0

# Initialize global surrogate
SURROGATE = SurrogateModel()

def update_operators_lib(op_name: str, delta_score: float):
    """Update operator library scores based on performance."""
    if op_name in OPERATORS_LIB:
        OPERATORS_LIB[op_name]['score'] += delta_score

# =============================================================================
# MAP-ELITES (Diversity Archive)
# Preserves diverse solutions across feature dimensions (Complexity x Depth)
# =============================================================================

class MAPElitesArchive:
    def __init__(self):
        # Grid: (len_bin, depth_bin) -> (score, Genome)
        self.grid: Dict[Tuple[int, int], Tuple[float, Genome]] = {}
        
    def _features(self, expr: str) -> Tuple[int, int]:
        """Map expression to feature bin coordinates."""
        # Dim 1: Complexity (Length): bins of 10 chars
        l = len(expr)
        l_bin = min(20, l // 10) 
        
        # Dim 2: Depth (nesting): bins of 1
        d = 0
        curr = 0
        for c in expr:
            if c == '(': curr += 1
            elif c == ')': curr -= 1
            d = max(d, curr)
        d_bin = min(10, d)
        
        return (l_bin, d_bin)
        
    def add(self, genome: Genome, score: float):
        """Add genome to grid if it's the best in its cell."""
        feat = self._features(genome.expr)
        # If cell empty or new score is better (lower is better in regression usually, but let's check score direction)
        # In this system, lower score is better (MSE/RMSE).
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

# Initialize global MAP-Elites
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
    return None

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

def crossover(rng: random.Random, a: str, b: str) -> str:
    ta = re.findall('\\w+|[+\\-*/().]|[\\d.]+', a)
    tb = re.findall('\\w+|[+\\-*/().]|[\\d.]+', b)
    if len(ta) < 3 or len(tb) < 3:
        return a
    i, j = (rng.randint(1, len(ta) - 1), rng.randint(1, len(tb) - 1))
    result = ''.join(ta[:i] + tb[j:])
    try:
        ast.parse(result, mode='eval')
        ok, _ = validate_expr(result)
        return result if ok else a
    except:
        return a

def seed_genome(rng: random.Random) -> Genome:
    choices = ['x', '(x*x)', '(x+1)', '(x-1)', '0.5', '1.0', 'sin(x)', 'cos(x)', 'tanh(x)', '(0.5*x+0.5)']
    return Genome(expr=rng.choice(choices))

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

@dataclass
class MetaState:
    op_weights: Dict[str, float] = field(default_factory=lambda: {k: 1.0 for k in OPERATORS})
    mutation_rate: float = 0.5859
    crossover_rate: float = 0.2
    complexity_lambda: float = 0.0001
    epsilon_explore: float = 0.15
    stuck_counter: int = 0

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

@dataclass
class Universe:
    uid: int
    seed: int
    meta: MetaState
    pool: List[Genome]
    library: FunctionLibrary
    best: Optional[Genome] = None
    best_score: float = float('inf')
    best_hold: float = float('inf')
    best_stress: float = float('inf')
    history: List[Dict] = field(default_factory=list)

    def step(self, gen: int, task: TaskSpec, pop_size: int) -> Dict:
        rng = random.Random(self.seed + gen * 1009)
        batch = sample_batch(rng, task)
        helpers = self.library.get_helpers()
        scored: List[Tuple[Genome, EvalResult]] = []
        for g in self.pool:
            res = evaluate(g, batch, self.meta.complexity_lambda)
            if res.ok:
                scored.append((g, res))
        if not scored:
            self.pool = [seed_genome(rng) for _ in range(pop_size)]
            return {'gen': gen, 'accepted': False, 'reason': 'reseed'}
        scored.sort(key=lambda t: t[1].score)
        
        # Add best to MAP-Elites
        if scored:
            best_g, best_res = scored[0]
            MAP_ELITES.add(best_g, best_res.score)

        elite_k = max(4, pop_size // 10)
        elites = [g for g, _ in scored[:elite_k]]
        children: List[Genome] = []
        for _ in range(pop_size - len(elites)):
            # Selection: Use elites or MAP-Elites for diversity
            if rng.random() < 0.1 and MAP_ELITES.grid:
                parent = MAP_ELITES.sample(rng) or rng.choice(elites)
            else:
                parent = rng.choice(elites)
            
            if rng.random() < self.meta.crossover_rate and len(elites) > 1:
                other = rng.choice(elites)
                new_expr = crossover(rng, parent.expr, other.expr)
                op_tag = 'crossover'
            elif rng.random() < self.meta.mutation_rate:
                use_synth = rng.random() < 0.3 and OPERATORS_LIB
                if use_synth:
                    synth_name = rng.choice(list(OPERATORS_LIB.keys()))
                    steps = OPERATORS_LIB[synth_name].get('steps', [])
                    new_expr = apply_synthesized_op(rng, parent.expr, steps)
                    op_tag = f'synth:{synth_name}'
                else:
                    op = self.meta.sample_op(rng)
                    if op in OPERATORS:
                        new_expr = OPERATORS[op](rng, parent.expr)
                    else:
                        new_expr = parent.expr
                    op_tag = f'mut:{op}'
            else:
                new_expr = parent.expr
                op_tag = 'copy'
            new_expr, used = self.library.maybe_inject(rng, new_expr)
            if used:
                op_tag += f'|lib:{used}'
            
            # RSI Intuition: Prune bad candidates using Surrogate prediction
            # If predicted error is high (> 2.0 or > 10x best), probibalistically revert
            if rng.random() < 0.4:
                pred = SURROGATE.predict(new_expr)
                if pred > max(0.5, self.best_score * 10):
                    new_expr = parent.expr
                    op_tag += "|pruned"

            children.append(Genome(expr=new_expr, parents=[parent.gid], op_tag=op_tag))
        self.pool = elites + children
        # Maybe evolve operators lib
        if rng.random() < 0.02:
            maybe_evolve_operators_lib(rng)
        if rng.random() < 0.05:
            self.library.maybe_adopt(rng, scored[0][0].expr)
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
        log = {'gen': gen, 'accepted': accepted, 'score': self.best_score, 'hold': self.best_hold, 'stress': self.best_stress, 'expr': self.best.expr if self.best else 'none'}
        self.history.append(log)
        
        # Train intuition
        if gen % 5 == 0:
            SURROGATE.train(self.history)
            
        return log

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
        print(f"[Gen {gen + 1:4d}] Score: {best.best_score:.4f} | Hold: {best.best_hold:.4f} | Stress: {best.best_stress:.4f} | {(best.best.expr if best.best else 'none')}")
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