import argparse
import ast
import copy
import json
import math
import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import pprint

META_DIR = Path(".rsi_meta")
RUNS_DIR = META_DIR / "runs"
ARCH_HISTORY_FILE = META_DIR / "arch_history.jsonl"
META_CONFIG_FILE = META_DIR / "meta_config.json"
DSL_REGISTRY_FILE = META_DIR / "dsl_registry.json"


def ensure_meta_dirs() -> None:
    """Ensure that persistence directories exist."""
    META_DIR.mkdir(exist_ok=True)
    RUNS_DIR.mkdir(exist_ok=True)


@dataclass
class SelfModPolicy:
    """Policy governing when self-modifications are accepted."""

    min_improvement_ratio: float = 0.98
    stability_window: int = 3
    overfit_penalty: float = 0.05
    acceptance_patience: int = 2

    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_improvement_ratio": self.min_improvement_ratio,
            "stability_window": self.stability_window,
            "overfit_penalty": self.overfit_penalty,
            "acceptance_patience": self.acceptance_patience,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "SelfModPolicy":
        return SelfModPolicy(
            min_improvement_ratio=data.get("min_improvement_ratio", 0.98),
            stability_window=data.get("stability_window", 3),
            overfit_penalty=data.get("overfit_penalty", 0.05),
            acceptance_patience=data.get("acceptance_patience", 2),
        )


@dataclass
class MetaConfig:
    """Meta-configuration for RSI engine."""

    population_size: int
    generations: int
    mutation_rate: float
    crossover_rate: float
    max_program_length: int
    max_depth: int
    registers: int
    complexity_weight: float
    runtime_weight: float
    architecture_mode: str
    dsl_operators: List[str]
    self_mod_policy: SelfModPolicy

    def to_dict(self) -> Dict[str, Any]:
        return {
            "population_size": self.population_size,
            "generations": self.generations,
            "mutation_rate": self.mutation_rate,
            "crossover_rate": self.crossover_rate,
            "max_program_length": self.max_program_length,
            "max_depth": self.max_depth,
            "registers": self.registers,
            "complexity_weight": self.complexity_weight,
            "runtime_weight": self.runtime_weight,
            "architecture_mode": self.architecture_mode,
            "dsl_operators": list(self.dsl_operators),
            "self_mod_policy": self.self_mod_policy.to_dict(),
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "MetaConfig":
        return MetaConfig(
            population_size=int(data.get("population_size", 40)),
            generations=int(data.get("generations", 12)),
            mutation_rate=float(data.get("mutation_rate", 0.2)),
            crossover_rate=float(data.get("crossover_rate", 0.4)),
            max_program_length=int(data.get("max_program_length", 14)),
            max_depth=int(data.get("max_depth", 4)),
            registers=int(data.get("registers", 4)),
            complexity_weight=float(data.get("complexity_weight", 0.01)),
            runtime_weight=float(data.get("runtime_weight", 0.001)),
            architecture_mode=data.get("architecture_mode", "linear_gp"),
            dsl_operators=list(data.get("dsl_operators", [])),
            self_mod_policy=SelfModPolicy.from_dict(data.get("self_mod_policy", {})),
        )


# @@METACONFIG_START@@
DEFAULT_META_CONFIG = {
    "population_size": 40,
    "generations": 12,
    "mutation_rate": 0.2,
    "crossover_rate": 0.4,
    "max_program_length": 14,
    "max_depth": 4,
    "registers": 4,
    "complexity_weight": 0.01,
    "runtime_weight": 0.001,
    "architecture_mode": "linear_gp",
    "dsl_operators": ["add", "sub", "mul", "neg", "abs", "square", "sin", "cos"],
    "self_mod_policy": {
        "min_improvement_ratio": 0.98,
        "stability_window": 3,
        "overfit_penalty": 0.05,
        "acceptance_patience": 2,
    },
}
# @@METACONFIG_END@@


# @@DSL_REGISTRY_START@@
DSL_OPERATORS = {
    "add": {"arity": 2, "kind": "base"},
    "sub": {"arity": 2, "kind": "base"},
    "mul": {"arity": 2, "kind": "base"},
    "neg": {"arity": 1, "kind": "base"},
    "abs": {"arity": 1, "kind": "base"},
    "square": {"arity": 1, "kind": "base"},
    "sin": {"arity": 1, "kind": "base"},
    "cos": {"arity": 1, "kind": "base"},
}
# @@DSL_REGISTRY_END@@


def load_meta_config() -> MetaConfig:
    """Load MetaConfig from disk or use defaults."""
    ensure_meta_dirs()
    if META_CONFIG_FILE.exists():
        data = json.loads(META_CONFIG_FILE.read_text())
        return MetaConfig.from_dict(data)
    return MetaConfig.from_dict(DEFAULT_META_CONFIG)


def save_meta_config(meta_config: MetaConfig) -> None:
    """Persist MetaConfig to disk."""
    ensure_meta_dirs()
    META_CONFIG_FILE.write_text(json.dumps(meta_config.to_dict(), indent=2))


def load_dsl_registry() -> Dict[str, Dict[str, Any]]:
    """Load DSL registry from disk or use defaults."""
    ensure_meta_dirs()
    if DSL_REGISTRY_FILE.exists():
        return json.loads(DSL_REGISTRY_FILE.read_text())
    return copy.deepcopy(DSL_OPERATORS)


def save_dsl_registry(registry: Dict[str, Dict[str, Any]]) -> None:
    """Persist DSL registry to disk."""
    ensure_meta_dirs()
    DSL_REGISTRY_FILE.write_text(json.dumps(registry, indent=2))


def sync_meta_config(meta_config: MetaConfig, registry: Dict[str, Dict[str, Any]]) -> MetaConfig:
    """Synchronize meta config operator list with registry keys."""
    meta_config.dsl_operators = sorted(registry.keys())
    return meta_config


def base_operator_impl(name: str, args: Sequence[float]) -> float:
    """Evaluate base operator by name."""
    if name == "add":
        return args[0] + args[1]
    if name == "sub":
        return args[0] - args[1]
    if name == "mul":
        return args[0] * args[1]
    if name == "neg":
        return -args[0]
    if name == "abs":
        return abs(args[0])
    if name == "square":
        return args[0] * args[0]
    if name == "sin":
        return math.sin(args[0])
    if name == "cos":
        return math.cos(args[0])
    raise ValueError(f"Unknown base operator {name}")


def apply_operator(name: str, args: Sequence[float], registry: Dict[str, Dict[str, Any]]) -> float:
    """Apply an operator from the DSL registry."""
    meta = registry.get(name)
    if not meta:
        raise ValueError(f"Unknown operator: {name}")
    if meta.get("kind") == "base":
        return base_operator_impl(name, args)
    if meta.get("kind") == "compose":
        value = args[0]
        for op_name in meta.get("sequence", []):
            value = apply_operator(op_name, [value], registry)
        return value
    raise ValueError(f"Unsupported operator kind: {meta.get('kind')}")


@dataclass
class Instruction:
    """Linear GP instruction."""

    op: str
    dest: int
    src_a: int
    src_b: Optional[int] = None


@dataclass
class LinearProgram:
    """Linear GP program representation."""

    instructions: List[Instruction]
    registers: int

    def execute_scalar(self, inputs: Sequence[float], registry: Dict[str, Dict[str, Any]]) -> float:
        regs = [0.0 for _ in range(self.registers)]
        for idx, value in enumerate(inputs[: self.registers]):
            regs[idx] = float(value)
        for instr in self.instructions:
            if instr.op not in registry:
                continue
            meta = registry[instr.op]
            if meta["arity"] == 1:
                regs[instr.dest] = apply_operator(instr.op, [regs[instr.src_a]], registry)
            else:
                rhs = regs[instr.src_b] if instr.src_b is not None else 0.0
                regs[instr.dest] = apply_operator(instr.op, [regs[instr.src_a], rhs], registry)
        return regs[0]

    def execute(self, input_value: Any, registry: Dict[str, Dict[str, Any]]) -> Any:
        if isinstance(input_value, list):
            return [self.execute_scalar([v], registry) for v in input_value]
        if isinstance(input_value, tuple):
            return self.execute_scalar(list(input_value), registry)
        return self.execute_scalar([input_value], registry)

    def complexity(self) -> int:
        return len(self.instructions)

    def mutate(self, meta_config: MetaConfig, registry: Dict[str, Dict[str, Any]], rng: random.Random) -> "LinearProgram":
        new_instructions = copy.deepcopy(self.instructions)
        if not new_instructions or rng.random() < 0.5:
            new_instructions.append(random_instruction(meta_config, registry, rng))
        else:
            idx = rng.randrange(len(new_instructions))
            new_instructions[idx] = random_instruction(meta_config, registry, rng)
        if len(new_instructions) > meta_config.max_program_length:
            new_instructions = new_instructions[: meta_config.max_program_length]
        return LinearProgram(new_instructions, self.registers)

    def crossover(self, other: "LinearProgram", rng: random.Random) -> "LinearProgram":
        if not self.instructions or not other.instructions:
            return copy.deepcopy(self)
        cut_a = rng.randrange(len(self.instructions))
        cut_b = rng.randrange(len(other.instructions))
        child_instr = self.instructions[:cut_a] + other.instructions[cut_b:]
        return LinearProgram(child_instr, self.registers)


@dataclass
class TreeNode:
    """Tree-based GP node."""

    op: Optional[str] = None
    value: Optional[float] = None
    children: List["TreeNode"] = field(default_factory=list)

    def evaluate(self, inputs: Sequence[float], registry: Dict[str, Dict[str, Any]]) -> float:
        if self.op is None:
            return self.value if self.value is not None else 0.0
        if self.op.startswith("input_"):
            index = int(self.op.split("_")[1])
            if index < len(inputs):
                return float(inputs[index])
            return 0.0
        meta = registry.get(self.op, {"arity": 1})
        if meta["arity"] == 1:
            return apply_operator(self.op, [self.children[0].evaluate(inputs, registry)], registry)
        return apply_operator(
            self.op,
            [
                self.children[0].evaluate(inputs, registry),
                self.children[1].evaluate(inputs, registry),
            ],
            registry,
        )

    def size(self) -> int:
        return 1 + sum(child.size() for child in self.children)


@dataclass
class TreeProgram:
    """Tree GP program representation."""

    root: TreeNode

    def execute(self, input_value: Any, registry: Dict[str, Dict[str, Any]]) -> Any:
        if isinstance(input_value, list):
            return [self.execute(v, registry) for v in input_value]
        if isinstance(input_value, tuple):
            return self.root.evaluate(list(input_value), registry)
        return self.root.evaluate([input_value], registry)

    def complexity(self) -> int:
        return self.root.size()

    def mutate(self, meta_config: MetaConfig, registry: Dict[str, Dict[str, Any]], rng: random.Random) -> "TreeProgram":
        new_root = copy.deepcopy(self.root)
        nodes = collect_nodes(new_root)
        if nodes:
            target = rng.choice(nodes)
            replacement = random_tree_node(meta_config, registry, rng, depth=0)
            target.op = replacement.op
            target.value = replacement.value
            target.children = replacement.children
        return TreeProgram(new_root)

    def crossover(self, other: "TreeProgram", rng: random.Random) -> "TreeProgram":
        new_root = copy.deepcopy(self.root)
        other_root = copy.deepcopy(other.root)
        nodes = collect_nodes(new_root)
        other_nodes = collect_nodes(other_root)
        if nodes and other_nodes:
            target = rng.choice(nodes)
            donor = rng.choice(other_nodes)
            target.op = donor.op
            target.value = donor.value
            target.children = donor.children
        return TreeProgram(new_root)


def collect_nodes(node: TreeNode) -> List[TreeNode]:
    nodes = [node]
    for child in node.children:
        nodes.extend(collect_nodes(child))
    return nodes


def random_tree_node(
    meta_config: MetaConfig, registry: Dict[str, Dict[str, Any]], rng: random.Random, depth: int
) -> TreeNode:
    if depth >= meta_config.max_depth or rng.random() < 0.3:
        if rng.random() < 0.5:
            return TreeNode(op=f"input_{rng.randrange(meta_config.registers)}")
        return TreeNode(op=None, value=rng.uniform(-2.0, 2.0))
    op = rng.choice(meta_config.dsl_operators)
    arity = registry.get(op, {"arity": 1})["arity"]
    children = [random_tree_node(meta_config, registry, rng, depth + 1) for _ in range(arity)]
    return TreeNode(op=op, children=children)


def random_instruction(meta_config: MetaConfig, registry: Dict[str, Dict[str, Any]], rng: random.Random) -> Instruction:
    op = rng.choice(meta_config.dsl_operators)
    arity = registry.get(op, {"arity": 1})["arity"]
    dest = rng.randrange(meta_config.registers)
    src_a = rng.randrange(meta_config.registers)
    src_b = rng.randrange(meta_config.registers) if arity == 2 else None
    return Instruction(op=op, dest=dest, src_a=src_a, src_b=src_b)


def random_program(meta_config: MetaConfig, registry: Dict[str, Dict[str, Any]], rng: random.Random) -> Any:
    if meta_config.architecture_mode == "tree_gp":
        return TreeProgram(random_tree_node(meta_config, registry, rng, depth=0))
    instructions = [
        random_instruction(meta_config, registry, rng)
        for _ in range(rng.randint(1, meta_config.max_program_length))
    ]
    return LinearProgram(instructions=instructions, registers=meta_config.registers)


@dataclass
class TaskSpec:
    """Specification of a learning task."""

    name: str
    generate_train_data: Callable[[int], List[Tuple[Any, Any]]]
    generate_holdout_data: Callable[[int], List[Tuple[Any, Any]]]


@dataclass
class EvaluationResult:
    """Evaluation metrics for a candidate program."""

    train_loss: float
    holdout_loss: float
    complexity: float
    runtime_penalty: float

    @property
    def total_score(self) -> float:
        return self.holdout_loss + self.train_loss + self.complexity + self.runtime_penalty


@dataclass
class SearchSummary:
    """Summary of a search run."""

    task_name: str
    best_score: float
    best_metrics: EvaluationResult
    best_program: Any
    history: List[float]


def loss_mse(outputs: List[float], targets: List[float]) -> float:
    if not outputs:
        return 1e6
    return statistics.mean((o - t) ** 2 for o, t in zip(outputs, targets))


def loss_sequence(outputs: List[List[float]], targets: List[List[float]]) -> float:
    if not outputs:
        return 1e6
    sample_losses = []
    for output, target in zip(outputs, targets):
        if not isinstance(output, list):
            sample_losses.append(1e6)
            continue
        n = min(len(output), len(target))
        if n == 0:
            sample_losses.append(1e6)
            continue
        mse = statistics.mean((output[i] - target[i]) ** 2 for i in range(n))
        length_penalty = abs(len(output) - len(target)) * 0.1
        sample_losses.append(mse + length_penalty)
    return statistics.mean(sample_losses)


def loss_classification(outputs: List[float], targets: List[int]) -> float:
    if not outputs:
        return 1.0
    preds = [1 if o >= 0.5 else 0 for o in outputs]
    errors = sum(1 for p, t in zip(preds, targets) if p != t)
    return errors / len(targets)


def evaluate_program(
    program: Any, task: TaskSpec, meta_config: MetaConfig, registry: Dict[str, Dict[str, Any]], seed: int
) -> EvaluationResult:
    train_data = task.generate_train_data(seed)
    holdout_data = task.generate_holdout_data(seed + 1)
    train_outputs = [program.execute(inp, registry) for inp, _ in train_data]
    holdout_outputs = [program.execute(inp, registry) for inp, _ in holdout_data]
    target_sample = train_data[0][1]
    if isinstance(target_sample, list):
        train_loss = loss_sequence(
            [list(map(float, v)) if isinstance(v, list) else [float(v)] for v in train_outputs],
            [list(map(float, t)) for _, t in train_data],
        )
        holdout_loss = loss_sequence(
            [list(map(float, v)) if isinstance(v, list) else [float(v)] for v in holdout_outputs],
            [list(map(float, t)) for _, t in holdout_data],
        )
    elif isinstance(target_sample, int):
        train_loss = loss_classification([float(v) for v in train_outputs], [int(t) for _, t in train_data])
        holdout_loss = loss_classification([float(v) for v in holdout_outputs], [int(t) for _, t in holdout_data])
    else:
        train_loss = loss_mse([float(v) for v in train_outputs], [float(t) for _, t in train_data])
        holdout_loss = loss_mse([float(v) for v in holdout_outputs], [float(t) for _, t in holdout_data])
    complexity = meta_config.complexity_weight * program.complexity()
    runtime_penalty = meta_config.runtime_weight * len(train_data) * program.complexity()
    return EvaluationResult(
        train_loss=train_loss,
        holdout_loss=holdout_loss,
        complexity=complexity,
        runtime_penalty=runtime_penalty,
    )


def selection(population: List[Any], scores: List[float], rng: random.Random) -> Any:
    total = sum(scores)
    if total == 0:
        return rng.choice(population)
    pick = rng.random() * total
    current = 0.0
    for program, score in zip(population, scores):
        current += score
        if current >= pick:
            return program
    return population[-1]


def run_algorithm_search(meta_config: MetaConfig, task: TaskSpec, seed: int) -> SearchSummary:
    """Run Level 0 algorithm search."""
    rng = random.Random(seed)
    registry = load_dsl_registry()
    population = [random_program(meta_config, registry, rng) for _ in range(meta_config.population_size)]
    history: List[float] = []
    best_program = None
    best_metrics = None
    best_score = float("inf")
    for _ in range(meta_config.generations):
        metrics = [evaluate_program(prog, task, meta_config, registry, seed) for prog in population]
        scores = [1.0 / (m.total_score + 1e-9) for m in metrics]
        for prog, metric in zip(population, metrics):
            if metric.total_score < best_score:
                best_score = metric.total_score
                best_program = prog
                best_metrics = metric
        history.append(best_score)
        next_population = []
        while len(next_population) < meta_config.population_size:
            parent_a = selection(population, scores, rng)
            if rng.random() < meta_config.crossover_rate:
                parent_b = selection(population, scores, rng)
                child = parent_a.crossover(parent_b, rng)
            else:
                child = copy.deepcopy(parent_a)
            if rng.random() < meta_config.mutation_rate:
                child = child.mutate(meta_config, registry, rng)
            next_population.append(child)
        population = next_population
    if best_program is None or best_metrics is None:
        best_program = population[0]
        best_metrics = metrics[0]
        best_score = best_metrics.total_score
    return SearchSummary(
        task_name=task.name,
        best_score=best_score,
        best_metrics=best_metrics,
        best_program=best_program,
        history=history,
    )


def task_sequence_double(seed: int) -> List[Tuple[List[float], List[float]]]:
    rng = random.Random(seed)
    data = []
    for _ in range(20):
        length = rng.randint(3, 6)
        seq = [rng.uniform(-3, 3) for _ in range(length)]
        target = [2 * v + 1 for v in seq]
        data.append((seq, target))
    return data


def task_sequence_double_holdout(seed: int) -> List[Tuple[List[float], List[float]]]:
    rng = random.Random(seed)
    data = []
    for _ in range(10):
        length = rng.randint(3, 6)
        seq = [rng.uniform(-3, 3) for _ in range(length)]
        target = [2 * v + 1 for v in seq]
        data.append((seq, target))
    return data


def task_symbolic_regression(seed: int) -> List[Tuple[Tuple[float, float], float]]:
    rng = random.Random(seed)
    data = []
    for _ in range(30):
        x = rng.uniform(-2, 2)
        y = rng.uniform(-2, 2)
        target = 1.5 * x - 0.8 * y + 0.3
        data.append(((x, y), target))
    return data


def task_symbolic_regression_holdout(seed: int) -> List[Tuple[Tuple[float, float], float]]:
    rng = random.Random(seed)
    data = []
    for _ in range(15):
        x = rng.uniform(-2, 2)
        y = rng.uniform(-2, 2)
        target = 1.5 * x - 0.8 * y + 0.3
        data.append(((x, y), target))
    return data


def task_classification(seed: int) -> List[Tuple[Tuple[float, float], int]]:
    rng = random.Random(seed)
    data = []
    for _ in range(30):
        x = rng.uniform(-1, 1)
        y = rng.uniform(-1, 1)
        target = 1 if x + y > 0 else 0
        data.append(((x, y), target))
    return data


def task_classification_holdout(seed: int) -> List[Tuple[Tuple[float, float], int]]:
    rng = random.Random(seed)
    data = []
    for _ in range(15):
        x = rng.uniform(-1, 1)
        y = rng.uniform(-1, 1)
        target = 1 if x + y > 0 else 0
        data.append(((x, y), target))
    return data


def build_tasks() -> Dict[str, TaskSpec]:
    return {
        "sequence_double": TaskSpec(
            name="sequence_double",
            generate_train_data=task_sequence_double,
            generate_holdout_data=task_sequence_double_holdout,
        ),
        "symbolic_regression": TaskSpec(
            name="symbolic_regression",
            generate_train_data=task_symbolic_regression,
            generate_holdout_data=task_symbolic_regression_holdout,
        ),
        "classify_plane": TaskSpec(
            name="classify_plane",
            generate_train_data=task_classification,
            generate_holdout_data=task_classification_holdout,
        ),
    }


def log_run(summary: SearchSummary, meta_config: MetaConfig, seed: int) -> None:
    ensure_meta_dirs()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    payload = {
        "task": summary.task_name,
        "seed": seed,
        "best_score": summary.best_score,
        "metrics": summary.best_metrics.__dict__,
        "history": summary.history,
        "meta_config": meta_config.to_dict(),
    }
    (RUNS_DIR / f"run_{timestamp}.json").write_text(json.dumps(payload, indent=2))


def perturb_meta_config(meta_config: MetaConfig, rng: random.Random) -> MetaConfig:
    new_config = copy.deepcopy(meta_config)
    new_config.population_size = max(10, int(new_config.population_size * rng.uniform(0.8, 1.2)))
    new_config.generations = max(4, int(new_config.generations * rng.uniform(0.7, 1.3)))
    new_config.mutation_rate = min(0.9, max(0.05, new_config.mutation_rate * rng.uniform(0.7, 1.3)))
    new_config.crossover_rate = min(0.9, max(0.1, new_config.crossover_rate * rng.uniform(0.7, 1.3)))
    new_config.complexity_weight = max(0.0001, new_config.complexity_weight * rng.uniform(0.7, 1.3))
    new_config.runtime_weight = max(0.0001, new_config.runtime_weight * rng.uniform(0.7, 1.3))
    return new_config


def run_meta_search(rounds: int, seed: int) -> MetaConfig:
    """Run Level 1 meta-parameter optimization."""
    rng = random.Random(seed)
    meta_config = load_meta_config()
    registry = load_dsl_registry()
    meta_config = sync_meta_config(meta_config, registry)
    tasks = list(build_tasks().values())
    best_score = float("inf")
    patience = 0
    for _ in range(rounds):
        candidate = perturb_meta_config(meta_config, rng)
        scores = []
        for task in tasks:
            summary = run_algorithm_search(candidate, task, rng.randint(0, 1_000_000))
            scores.append(summary.best_score)
        avg_score = statistics.mean(scores)
        if avg_score < best_score * meta_config.self_mod_policy.min_improvement_ratio:
            best_score = avg_score
            meta_config = candidate
            patience = 0
            save_meta_config(meta_config)
        else:
            patience += 1
            if patience >= meta_config.self_mod_policy.acceptance_patience:
                break
    return meta_config


def extract_operator_patterns(programs: List[Any], registry: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    """Extract frequent operator patterns from programs."""
    counts: Dict[str, int] = {}
    for prog in programs:
        if isinstance(prog, LinearProgram):
            for idx in range(len(prog.instructions) - 1):
                first = prog.instructions[idx].op
                second = prog.instructions[idx + 1].op
                if registry.get(first, {}).get("arity") != 1 or registry.get(second, {}).get("arity") != 1:
                    continue
                pair = (first, second)
                key = "->".join(pair)
                counts[key] = counts.get(key, 0) + 1
    return counts


def evolve_dsl(registry: Dict[str, Dict[str, Any]], programs: List[Any]) -> Dict[str, Dict[str, Any]]:
    """Level 2: evolve DSL operators based on program patterns."""
    counts = extract_operator_patterns(programs, registry)
    if not counts:
        return registry
    candidate = max(counts.items(), key=lambda item: item[1])
    op_seq = candidate[0].split("->")
    new_name = f"compose_{op_seq[0]}_{op_seq[1]}"
    if new_name not in registry:
        registry[new_name] = {"arity": 1, "kind": "compose", "sequence": op_seq}
    save_dsl_registry(registry)
    return registry


def run_architecture_search(rounds: int, seed: int) -> MetaConfig:
    """Level 3: architecture evolution."""
    ensure_meta_dirs()
    rng = random.Random(seed)
    meta_config = load_meta_config()
    registry = load_dsl_registry()
    meta_config = sync_meta_config(meta_config, registry)
    tasks = list(build_tasks().values())
    candidates = ["linear_gp", "tree_gp"]
    for _ in range(rounds):
        results = []
        for mode in candidates:
            candidate_config = copy.deepcopy(meta_config)
            candidate_config.architecture_mode = mode
            scores = []
            for task in tasks:
                summary = run_algorithm_search(candidate_config, task, rng.randint(0, 1_000_000))
                scores.append(summary.best_score)
            avg_score = statistics.mean(scores)
            results.append((avg_score, mode))
            with ARCH_HISTORY_FILE.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"mode": mode, "score": avg_score, "time": time.time()}) + "\n")
        best_score, best_mode = min(results, key=lambda item: item[0])
        if best_score < float("inf"):
            meta_config.architecture_mode = best_mode
            save_meta_config(meta_config)
    return meta_config


def perturb_policy(policy: SelfModPolicy, rng: random.Random) -> SelfModPolicy:
    new_policy = copy.deepcopy(policy)
    new_policy.min_improvement_ratio = min(1.0, max(0.8, policy.min_improvement_ratio * rng.uniform(0.95, 1.05)))
    new_policy.stability_window = max(2, int(policy.stability_window + rng.choice([-1, 0, 1])))
    new_policy.overfit_penalty = max(0.0, policy.overfit_penalty * rng.uniform(0.8, 1.2))
    new_policy.acceptance_patience = max(1, int(policy.acceptance_patience + rng.choice([-1, 0, 1])))
    return new_policy


def run_policy_evolution(rounds: int, seed: int) -> MetaConfig:
    """Level 4: evolve self-modification policy."""
    rng = random.Random(seed)
    meta_config = load_meta_config()
    registry = load_dsl_registry()
    meta_config = sync_meta_config(meta_config, registry)
    tasks = list(build_tasks().values())
    best_score = float("inf")
    for _ in range(rounds):
        candidate_policy = perturb_policy(meta_config.self_mod_policy, rng)
        candidate_config = copy.deepcopy(meta_config)
        candidate_config.self_mod_policy = candidate_policy
        scores = []
        for task in tasks:
            summary = run_algorithm_search(candidate_config, task, rng.randint(0, 1_000_000))
            scores.append(summary.best_score)
        avg_score = statistics.mean(scores)
        if avg_score < best_score * candidate_policy.min_improvement_ratio:
            best_score = avg_score
            meta_config.self_mod_policy = candidate_policy
            save_meta_config(meta_config)
    return meta_config


def run_self_rewrite(meta_config: MetaConfig, registry: Dict[str, Dict[str, Any]], dry_run: bool = True) -> str:
    """Level 5: mechanically rewrite delimited blocks in this file."""
    source_path = Path("rsi_engine.py")
    source = source_path.read_text()
    meta_block = "# @@METACONFIG_START@@\n" + f"DEFAULT_META_CONFIG = {pprint.pformat(meta_config.to_dict(), indent=4)}\n" + "# @@METACONFIG_END@@"
    dsl_block = "# @@DSL_REGISTRY_START@@\n" + f"DSL_OPERATORS = {pprint.pformat(registry, indent=4)}\n" + "# @@DSL_REGISTRY_END@@"

    def replace_block(text: str, start: str, end: str, new_block: str) -> str:
        start_idx = text.find(start)
        end_idx = text.find(end)
        if start_idx == -1 or end_idx == -1:
            raise ValueError("Delimiters not found for block replacement")
        end_idx += len(end)
        return text[:start_idx] + new_block + text[end_idx:]

    updated = replace_block(source, "# @@METACONFIG_START@@", "# @@METACONFIG_END@@", meta_block)
    updated = replace_block(updated, "# @@DSL_REGISTRY_START@@", "# @@DSL_REGISTRY_END@@", dsl_block)
    try:
        ast.parse(updated)
    except SyntaxError as exc:
        raise ValueError(f"Updated source failed to parse: {exc}")
    if dry_run:
        return "dry_run: updated blocks prepared"
    source_path.write_text(updated)
    return "rewrite applied"


def run_rsi_loop(rounds: int, seed: int) -> None:
    """Run the high-level RSI loop."""
    rng = random.Random(seed)
    meta_config = load_meta_config()
    registry = load_dsl_registry()
    meta_config = sync_meta_config(meta_config, registry)
    tasks = build_tasks()
    for _ in range(rounds):
        task = rng.choice(list(tasks.values()))
        summary = run_algorithm_search(meta_config, task, rng.randint(0, 1_000_000))
        log_run(summary, meta_config, seed)
        registry = evolve_dsl(registry, [summary.best_program])
        save_dsl_registry(registry)
        meta_config = sync_meta_config(meta_config, registry)
        meta_config = run_meta_search(1, rng.randint(0, 1_000_000))
        meta_config = run_architecture_search(1, rng.randint(0, 1_000_000))
        meta_config = run_policy_evolution(1, rng.randint(0, 1_000_000))
        if rng.random() < 0.2:
            run_self_rewrite(meta_config, registry, dry_run=True)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Recursive Self-Improvement Engine")
    subparsers = parser.add_subparsers(dest="command", required=True)

    solve_parser = subparsers.add_parser("solve")
    solve_parser.add_argument("--generations", type=int, default=0)
    solve_parser.add_argument("--task", type=str, required=True)
    solve_parser.add_argument("--seed", type=int, default=0)

    meta_parser = subparsers.add_parser("meta-search")
    meta_parser.add_argument("--rounds", type=int, default=3)
    meta_parser.add_argument("--seed", type=int, default=0)

    arch_parser = subparsers.add_parser("arch-search")
    arch_parser.add_argument("--rounds", type=int, default=2)
    arch_parser.add_argument("--seed", type=int, default=0)

    loop_parser = subparsers.add_parser("rsi-loop")
    loop_parser.add_argument("--rounds", type=int, default=3)
    loop_parser.add_argument("--seed", type=int, default=0)

    rewrite_parser = subparsers.add_parser("self-rewrite")
    rewrite_parser.add_argument("--apply", action="store_true")

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Main entry point."""
    args = parse_args(argv)
    ensure_meta_dirs()
    meta_config = load_meta_config()
    registry = load_dsl_registry()
    meta_config = sync_meta_config(meta_config, registry)
    tasks = build_tasks()

    if args.command == "solve":
        task = tasks.get(args.task)
        if not task:
            raise ValueError(f"Unknown task: {args.task}")
        if args.generations:
            meta_config.generations = args.generations
        summary = run_algorithm_search(meta_config, task, args.seed)
        log_run(summary, meta_config, args.seed)
        print(json.dumps({"task": summary.task_name, "best_score": summary.best_score}, indent=2))
    elif args.command == "meta-search":
        meta_config = run_meta_search(args.rounds, args.seed)
        print(json.dumps(meta_config.to_dict(), indent=2))
    elif args.command == "arch-search":
        meta_config = run_architecture_search(args.rounds, args.seed)
        print(json.dumps(meta_config.to_dict(), indent=2))
    elif args.command == "rsi-loop":
        run_rsi_loop(args.rounds, args.seed)
        print(json.dumps({"status": "completed", "rounds": args.rounds}, indent=2))
    elif args.command == "self-rewrite":
        result = run_self_rewrite(meta_config, registry, dry_run=not args.apply)
        print(json.dumps({"status": result}, indent=2))


if __name__ == "__main__":
    main()
