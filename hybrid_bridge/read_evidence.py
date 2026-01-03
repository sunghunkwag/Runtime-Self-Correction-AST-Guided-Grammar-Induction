"""Streaming JSONL reader for Omega evidence."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterator, Optional

from .schema import EvidenceRecord


def _coerce_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_str(*values: Any) -> Optional[str]:
    for value in values:
        if isinstance(value, str) and value:
            return value
    return None


def _extract_subseq(payload: Dict[str, Any]) -> tuple[Optional[list[str]], Optional[str]]:
    subseq = payload.get("subseq")
    subseq_str = payload.get("subseq_str")
    if isinstance(subseq, list):
        try:
            return [str(item) for item in subseq], subseq_str if isinstance(subseq_str, str) else None
        except Exception:
            return None, subseq_str if isinstance(subseq_str, str) else None
    if isinstance(subseq_str, str):
        return None, subseq_str
    return None, None


def stream_evidence(path: str) -> Iterator[EvidenceRecord]:
    """Yield EvidenceRecord entries from a JSONL file.

    Lines that are not valid JSON or are non-evidence headers are skipped.
    Missing fields are tolerated and set to None.
    """

    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and payload.get("type") == "header":
                continue
            if not isinstance(payload, dict):
                continue

            metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
            diag = payload.get("diag") if isinstance(payload.get("diag"), dict) else {}
            reasons = payload.get("reasons")
            pass_reason = None
            if isinstance(reasons, list) and reasons:
                pass_reason = ";".join(str(item) for item in reasons)
            elif isinstance(payload.get("pass_reason"), str):
                pass_reason = payload.get("pass_reason")

            subseq, subseq_str = _extract_subseq(payload)
            if subseq is None and subseq_str is None:
                subseq, subseq_str = _extract_subseq(diag)

            record = EvidenceRecord(
                genome_id=_first_str(payload.get("genome_id"), payload.get("gid")),
                generation=_coerce_int(payload.get("generation", payload.get("gen"))),
                cfg_hash=_first_str(payload.get("cfg_hash"), diag.get("cfg_hash")),
                loops_count=_coerce_int(
                    payload.get("loops_count", metrics.get("loops"))
                ),
                scc_count=_coerce_int(payload.get("scc_count", metrics.get("scc_n"))),
                coverage=_coerce_float(payload.get("coverage", metrics.get("coverage"))),
                subseq=subseq,
                subseq_str=subseq_str,
                repro_signature=_first_str(payload.get("repro_signature"), diag.get("success_hash")),
                pass_reason=pass_reason,
                extra={"raw": payload},
            )
            yield record
