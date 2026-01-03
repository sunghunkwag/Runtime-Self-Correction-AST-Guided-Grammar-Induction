"""Dataclasses for hybrid bridge evidence and config schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EvidenceRecord:
    """Normalized evidence record parsed from Omega JSONL output."""

    genome_id: Optional[str] = None
    generation: Optional[int] = None
    cfg_hash: Optional[str] = None
    loops_count: Optional[int] = None
    scc_count: Optional[int] = None
    coverage: Optional[float] = None
    subseq: Optional[List[str]] = None
    subseq_str: Optional[str] = None
    repro_signature: Optional[str] = None
    pass_reason: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OmegaConfig:
    """Omega detector configuration handoff schema."""

    K_initial: int = 5
    L_initial: int = 8
    C_coverage: float = 0.55
    f_rarity: float = 0.001
    N_reproducibility: int = 4
    require_both: bool = True
    min_loops: int = 1
    min_scc: int = 1
    warmup_generations: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "K_initial": self.K_initial,
            "L_initial": self.L_initial,
            "C_coverage": self.C_coverage,
            "f_rarity": self.f_rarity,
            "N_reproducibility": self.N_reproducibility,
            "require_both": self.require_both,
            "min_loops": self.min_loops,
            "min_scc": self.min_scc,
        }
        if self.warmup_generations is not None:
            data["warmup_generations"] = self.warmup_generations
        data.update(self.extra)
        return data

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "OmegaConfig":
        extra = dict(payload)
        known = {}
        for key in [
            "K_initial",
            "L_initial",
            "C_coverage",
            "f_rarity",
            "N_reproducibility",
            "require_both",
            "min_loops",
            "min_scc",
            "warmup_generations",
        ]:
            if key in extra:
                known[key] = extra.pop(key)
        return cls(
            K_initial=int(known.get("K_initial", 5)),
            L_initial=int(known.get("L_initial", 8)),
            C_coverage=float(known.get("C_coverage", 0.55)),
            f_rarity=float(known.get("f_rarity", 0.001)),
            N_reproducibility=int(known.get("N_reproducibility", 4)),
            require_both=bool(known.get("require_both", True)),
            min_loops=int(known.get("min_loops", 1)),
            min_scc=int(known.get("min_scc", 1)),
            warmup_generations=(
                int(known["warmup_generations"]) if "warmup_generations" in known else None
            ),
            extra=extra,
        )
