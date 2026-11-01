"""JSONL utilities: append-only, resume-safe, last-wins by key."""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator

from ctm.types import result_key


def append_jsonl(path: str | Path, row: Dict[str, Any]) -> None:
    """Append one JSONL row with an exclusive file lock (multi-GPU safe)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False, default=_default) + "\n"
    with path.open("a", encoding="utf-8") as f:
        try:
            import fcntl

            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception:
            f.write(line)
            f.flush()


def _default(obj: Any) -> Any:
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            pass
    return str(obj)


def iter_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return
        yield  # pragma: no cover
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_jsonl_last_wins(
    path: str | Path,
    key_fields: Iterable[str] = ("utterance_id", "codec", "condition"),
) -> Dict[str, Dict[str, Any]]:
    """Keep the last row for each composite key."""
    out: Dict[str, Dict[str, Any]] = {}
    for row in iter_jsonl(path):
        key = "||".join(str(row.get(k, "")) for k in key_fields)
        out[key] = row
    return out


def completed_keys(path: str | Path) -> set[str]:
    """Keys that finished with status ok (successful rows only for resume skip)."""
    done = set()
    for key, row in load_jsonl_last_wins(path).items():
        if row.get("status") == "ok":
            done.add(key)
    return done


def make_key(utterance_id: str, codec: str, condition: str) -> str:
    return result_key(utterance_id, codec, condition)
