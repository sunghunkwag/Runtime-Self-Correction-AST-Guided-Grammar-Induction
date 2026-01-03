"""Minimal self-test for the hybrid bridge."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from .read_evidence import stream_evidence
from .run_round import run_round, build_cli
from .schema import OmegaConfig
from .tuner import tune_config


def _write_fake_evidence(path: Path) -> None:
    records = [
        {
            "genome_id": "g1",
            "generation": 1,
            "cfg_hash": "cfg_a",
            "loops_count": 2,
            "scc_count": 1,
            "coverage": 0.45,
            "subseq": ["ADD", "JMP"],
            "repro_signature": "r1",
            "pass_reason": "ok",
        },
        {
            "genome_id": "g2",
            "generation": 2,
            "cfg_hash": "cfg_b",
            "loops_count": 3,
            "scc_count": 2,
            "coverage": 0.55,
            "subseq_str": "MOV,ADD",
            "repro_signature": "r2",
            "pass_reason": "ok",
        },
        {
            "genome_id": "g3",
            "generation": 3,
            "cfg_hash": "cfg_b",
            "loops_count": 1,
            "scc_count": 1,
            "coverage": 0.65,
            "subseq": ["CALL", "RET"],
            "repro_signature": "r3",
            "pass_reason": "ok",
        },
    ]
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        evidence_path = tmp_path / "evidence.jsonl"
        config_path = tmp_path / "omega_config.json"
        _write_fake_evidence(evidence_path)

        parsed = list(stream_evidence(str(evidence_path)))
        assert len(parsed) == 3, "parser should read 3 evidence lines"

        tuning = tune_config(parsed, OmegaConfig(), max_generations=10, seed=1)
        with config_path.open("w", encoding="utf-8") as handle:
            json.dump(tuning.config.to_dict(), handle)
        assert config_path.exists(), "omega_config.json should be created"

        args = build_cli().parse_args(
            [
                "--dry-run",
                "--out",
                str(evidence_path),
                "--config-out",
                str(config_path),
                "--max_generations",
                "10",
            ]
        )
        summary = run_round(args)
        assert summary.evidence_count == 3, "dry-run should read evidence"

    print("SELFTEST_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
