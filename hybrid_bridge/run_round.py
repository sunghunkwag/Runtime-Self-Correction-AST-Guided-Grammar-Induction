"""Orchestrate Omega Forge runs and config tuning."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List

from .read_evidence import stream_evidence
from .schema import OmegaConfig
from .tuner import TuningResult, tune_config, write_config


@dataclass
class RoundSummary:
    generations: int
    evidence_count: int
    unique_cfg_hashes: int
    avg_loops: float
    avg_coverage: float
    tuner_actions: List[str]


def _summarize(evidence_path: Path) -> tuple[int, int, int, float, float]:
    loops_total = 0
    coverage_total = 0.0
    count = 0
    cfg_hashes = set()
    max_generation = 0

    for record in stream_evidence(str(evidence_path)):
        count += 1
        if record.generation is not None:
            max_generation = max(max_generation, record.generation)
        if record.loops_count is not None:
            loops_total += record.loops_count
        if record.coverage is not None:
            coverage_total += record.coverage
        if record.cfg_hash:
            cfg_hashes.add(record.cfg_hash)

    avg_loops = loops_total / count if count else 0.0
    avg_coverage = coverage_total / count if count else 0.0
    return max_generation, count, len(cfg_hashes), avg_loops, avg_coverage


def run_round(args: argparse.Namespace) -> RoundSummary:
    evidence_path = Path(args.out)
    omega_config_path = Path(args.config_out)

    if not args.dry_run:
        cmd = [
            "python",
            args.omega_path,
            "evidence_run",
            "--target",
            str(args.target),
            "--max_generations",
            str(args.max_generations),
            "--out",
            str(evidence_path),
            "--seed",
            str(args.seed),
        ]
        subprocess.run(cmd, check=True)

    base_config = OmegaConfig()
    if args.base_config:
        base_payload = json.loads(Path(args.base_config).read_text(encoding="utf-8"))
        base_config = OmegaConfig.from_dict(base_payload)

    evidence_records = list(stream_evidence(str(evidence_path)))
    tuning: TuningResult = tune_config(
        evidence_records,
        base_config=base_config,
        max_generations=args.max_generations,
        seed=args.seed,
    )
    write_config(str(omega_config_path), tuning.config)

    generations, evidence_count, unique_cfg_hashes, avg_loops, avg_coverage = _summarize(
        evidence_path
    )

    summary = RoundSummary(
        generations=generations,
        evidence_count=evidence_count,
        unique_cfg_hashes=unique_cfg_hashes,
        avg_loops=avg_loops,
        avg_coverage=avg_coverage,
        tuner_actions=tuning.actions,
    )
    return summary


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hybrid bridge run orchestrator")
    parser.add_argument("--omega-path", default="OMEGA_FORGE_V13.py")
    parser.add_argument("--target", type=int, default=6)
    parser.add_argument("--max_generations", type=int, default=2000)
    parser.add_argument("--out", default="evidence.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config-out", default="omega_config.json")
    parser.add_argument("--base-config", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_cli().parse_args()
    summary = run_round(args)
    print(
        "summary"
        f" generations={summary.generations}"
        f" evidence_count={summary.evidence_count}"
        f" unique_cfg_hashes={summary.unique_cfg_hashes}"
        f" avg_loops={summary.avg_loops:.2f}"
        f" avg_coverage={summary.avg_coverage:.3f}"
        f" tuner_actions={','.join(summary.tuner_actions)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
