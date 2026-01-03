"""Simple deterministic tuner for Omega parameters."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, List

from .schema import EvidenceRecord, OmegaConfig


@dataclass
class TuningResult:
    config: OmegaConfig
    actions: List[str]


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def tune_config(
    evidence: Iterable[EvidenceRecord],
    base_config: OmegaConfig,
    max_generations: int,
    seed: int = 0,
) -> TuningResult:
    """Tune OmegaConfig based on evidence statistics."""

    records = list(evidence)
    evidence_count = len(records)
    actions: List[str] = [
        f"seed={seed}",
        f"evidence_count={evidence_count}",
    ]

    cfg = OmegaConfig.from_dict(base_config.to_dict())
    unique_cfg_hashes = {r.cfg_hash for r in records if r.cfg_hash}
    repeated_cfg = len(unique_cfg_hashes) < evidence_count if evidence_count else False

    if evidence_count == 0:
        cfg.K_initial = max(2, cfg.K_initial - 1)
        cfg.L_initial = max(6, cfg.L_initial - 1)
        cfg.C_coverage = _clamp(cfg.C_coverage - 0.05, 0.35, 0.95)
        actions.append("relax_constraints: no evidence")
        if cfg.warmup_generations:
            cfg.require_both = False
            actions.append("warmup: require_both disabled")
    else:
        threshold = max(1, int(max_generations * 0.1))
        if evidence_count > threshold:
            cfg.K_initial += 1
            cfg.L_initial += 1
            cfg.C_coverage = _clamp(cfg.C_coverage + 0.05, 0.35, 0.95)
            cfg.require_both = True
            actions.append("tighten_constraints: high evidence rate")
        else:
            actions.append("no_constraint_change")

    if repeated_cfg:
        cfg.extra["ban_repeats"] = True
        actions.append("ban_repeats enabled (Omega support may be required)")

    if cfg.require_both:
        cfg.min_loops = max(1, cfg.min_loops)
        cfg.min_scc = max(1, cfg.min_scc)
        actions.append("strict_mode: enforced min_loops/min_scc")

    return TuningResult(config=cfg, actions=actions)


def write_config(path: str, config: OmegaConfig) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")
