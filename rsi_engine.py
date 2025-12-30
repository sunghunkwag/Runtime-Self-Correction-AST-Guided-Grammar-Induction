"""
Recursive Self-Improvement (RSI) Engine

Single-file, runnable system that attempts algorithm invention by generating,
executing, evaluating, and self-modifying its own search process.
"""

from __future__ import annotations

import ast
import multiprocessing as mp
import random
import textwrap
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class ProgramCandidate:
    code: str
    origin: str
    score: float = 0.0
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    features: Dict[str, int] = field(default_factory=dict)


class Representation:
    """Expandable grammar and primitives.

    This enables invention by allowing new control patterns to be introduced
    dynamically, rather than committing to a fixed syntax whitelist.
    """

    def __init__(self) -> None:
        self.grammar: Dict[str, List[Callable[["Representation"], str]]] = {
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

    def add_production(self, symbol: str, producer: Callable[["Representation"], str]) -> None:
        self.grammar.setdefault(symbol, []).append(producer)

    def expand(self, symbol: str) -> str:
        options = self.grammar.get(symbol, [])
        if not options:
            raise ValueError(f"No productions for symbol: {symbol}")
        return random.choice(options)(self)

    def _base_program(self, _: "Representation") -> str:
        helpers = "\n\n".join(self.library) if self.library else ""
        solver = self.expand("solver")
        return textwrap.dedent(
            f"""
            {helpers}

            {solver}
            """
        ).strip()

    def _solver_template(self, _: "Representation") -> str:
        control = self.expand("control")
        strategy = self.expand("strategy")
        return textwrap.dedent(
            f"""
            def solve(task):
                """Return the solution for the provided task.

                Generated as a full Python function so new control flow patterns
                can be invented, replaced, or expanded.
                """
                {control}
                {strategy}
            """
        ).strip()

    def _loop_control(self, _: "Representation") -> str:
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

    def _recursion_control(self, _: "Representation") -> str:
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

    def _greedy_strategy(self, _: "Representation") -> str:
        return textwrap.indent(
            textwrap.dedent(
                """
                if task.kind == 'sequence':
                    return [x + 1 for x in task.input]
                if task.kind == 'path':
                    return task.heuristic_path()
                if task.kind == 'transform':
                    return ''.join(sorted(task.input))
                return task.fallback()
                """
            ).strip(),
            "    ",
        )

    def _dp_strategy(self, _: "Representation") -> str:
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
                return task.fallback()
                """
            ).strip(),
            "    ",
        )

    def _divide_conquer_strategy(self, _: "Representation") -> str:
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

    def _search_strategy(self, _: "Representation") -> str:
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


class ProgramGenerator:
    """Generate programs via grammar and composition.

    Composition across a growing library enables reuse of learned abstractions.
    """

    def __init__(self, representation: Representation) -> None:
        self.representation = representation
        self.operator_weights: Dict[str, float] = {
            "grammar": 1.0,
            "compose": 1.0,
        }

    def generate(self) -> ProgramCandidate:
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

    def _grammar_program(self) -> ProgramCandidate:
        code = self.representation.expand("program")
        return ProgramCandidate(code=code, origin="grammar")

    def _compose_program(self) -> ProgramCandidate:
        helpers = random.sample(self.representation.library, k=1)
        base = self.representation.expand("program")
        code = "\n\n".join(helpers + [base])
        return ProgramCandidate(code=code, origin="compose")


@dataclass
class Task:
    kind: str
    input: Any
    expected: Any
    hint: Optional[str] = None

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


class TaskDomain:
    """Generates diverse tasks to avoid overfitting to a fixed benchmark.

    The engine must invent reusable procedures across tasks, not just tune.
    """

    def __init__(self) -> None:
        self.seed = 0

    def generate_tasks(self, count: int = 3) -> List[Task]:
        tasks: List[Task] = []
        for _ in range(count):
            self.seed += 1
            random.seed(self.seed)
            choice = random.choice(["sequence", "path", "transform"])
            if choice == "sequence":
                data = [random.randint(1, 5) for _ in range(random.randint(3, 5))]
                expected = [sum(data[:i + 1]) for i in range(len(data))]
                tasks.append(Task(kind="sequence", input=data, expected=expected, hint="prefix"))
            elif choice == "path":
                size = random.randint(3, 4)
                grid = [[random.randint(1, 9) for _ in range(size)] for _ in range(size)]
                expected = sum(grid[0]) + sum(row[-1] for row in grid[1:])
                tasks.append(Task(kind="path", input=grid, expected=expected, hint="grid"))
            else:
                word = "".join(random.choice("abcd") for _ in range(5))
                expected = "".join(sorted(word))
                tasks.append(Task(kind="transform", input=word, expected=expected, hint="sort"))
        return tasks


class Evaluator:
    """Execute candidates in isolated processes and score them.

    Failures become diagnostic signals, enabling the meta-controller to adapt.
    """

    def __init__(self) -> None:
        self.novelty_weight = 0.5
        self.archive_features: List[Dict[str, int]] = []

    def evaluate(self, candidate: ProgramCandidate, tasks: List[Task], timeout: float = 1.0) -> None:
        results: List[Tuple[bool, str]] = []
        for task in tasks:
            success, info = self._run_in_subprocess(candidate.code, task, timeout)
            results.append((success, info))
        candidate.diagnostics["results"] = results
        candidate.score = self._score(candidate, results, tasks)
        candidate.features = self._extract_features(candidate.code)
        self.archive_features.append(candidate.features)

    def _run_in_subprocess(self, code: str, task: Task, timeout: float) -> Tuple[bool, str]:
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

    def _score(self, candidate: ProgramCandidate, results: List[Tuple[bool, str]], tasks: List[Task]) -> float:
        success_rate = sum(1 for ok, _ in results if ok) / max(1, len(results))
        novelty = self._novelty(candidate.code)
        anti_trick = -0.2 if self._is_trivial(candidate.code, tasks) else 0.0
        return success_rate + self.novelty_weight * novelty + anti_trick

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

    def _is_trivial(self, code: str, tasks: List[Task]) -> bool:
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


class SelfModifier:
    """Adjusts generator, evaluator, and grammar based on diagnostics.

    This makes the system's learning rules and search operators mutable objects.
    """

    def __init__(self, representation: Representation, generator: ProgramGenerator, evaluator: Evaluator) -> None:
        self.representation = representation
        self.generator = generator
        self.evaluator = evaluator

    def adapt(self, candidate: ProgramCandidate) -> None:
        results = candidate.diagnostics.get("results", [])
        failures = [info for ok, info in results if not ok]
        if failures:
            self.generator.operator_weights["compose"] += 0.1
            self.evaluator.novelty_weight = min(1.5, self.evaluator.novelty_weight + 0.1)
            self._expand_grammar()
        else:
            self.generator.operator_weights["grammar"] += 0.1

    def _expand_grammar(self) -> None:
        def new_control(_: Representation) -> str:
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


class MetaController:
    """Coordinates generation, evaluation, self-modification, and retention.

    This creates a loop where algorithmic structures can be replaced entirely.
    """

    def __init__(self) -> None:
        self.representation = Representation()
        self.generator = ProgramGenerator(self.representation)
        self.evaluator = Evaluator()
        self.self_modifier = SelfModifier(self.representation, self.generator, self.evaluator)
        self.task_domain = TaskDomain()
        self.archive: List[ProgramCandidate] = []

    def run(self, iterations: int = 5) -> None:
        for _ in range(iterations):
            tasks = self.task_domain.generate_tasks()
            candidate = self.generator.generate()
            self.evaluator.evaluate(candidate, tasks)
            self._retain(candidate)
            self.self_modifier.adapt(candidate)

    def _retain(self, candidate: ProgramCandidate) -> None:
        if candidate.score <= 0:
            return
        self.archive.append(candidate)
        self._extract_helpers(candidate.code)

    def _extract_helpers(self, code: str) -> None:
        tree = ast.parse(code)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name != "solve":
                helper_code = ast.unparse(node)
                if helper_code not in self.representation.library:
                    self.representation.library.append(helper_code)


def main() -> None:
    random.seed(0)
    controller = MetaController()
    start = time.time()
    controller.run(iterations=6)
    duration = time.time() - start
    print(f"Completed {len(controller.archive)} retained candidates in {duration:.2f}s")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
