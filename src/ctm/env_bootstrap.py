"""Load project .env and normalize Hugging Face credentials (no secret logging)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

_ROOT = Path(__file__).resolve().parents[2]
_TOKEN_ALIASES = (
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACE_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
)


def _parse_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = val
        elif key and not os.environ.get(key):
            os.environ[key] = val


def _first_nonempty(names: Iterable[str]) -> Optional[str]:
    for name in names:
        v = os.environ.get(name)
        if v:
            return v
    return None


def load_project_env(*, quiet: bool = True) -> bool:
    """
    Load ``.env`` from the repo root and map token aliases to ``HF_TOKEN``.
    Returns True if an HF token is available afterwards.
    """
    env_path = Path(os.environ.get("CTM_ENV_FILE", str(_ROOT / ".env")))
    _parse_env_file(env_path)
    token = _first_nonempty(_TOKEN_ALIASES)
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", token)
    # Optional lab proxy: only applied when explicitly requested via CTM_DEFAULT_PROXY
    # (a hard-coded default silently broke hub access on machines outside that network).
    proxy = os.environ.get("CTM_DEFAULT_PROXY")
    if proxy:
        os.environ.setdefault("http_proxy", proxy)
        os.environ.setdefault("https_proxy", proxy)
        os.environ.setdefault("HTTP_PROXY", proxy)
        os.environ.setdefault("HTTPS_PROXY", proxy)
    if not quiet:
        if token:
            print(f"[ctm-env] HF auth: ON (token length={len(token)})")
        else:
            print("[ctm-env] HF auth: OFF — set HF_TOKEN in .env")
    return bool(token)


# Auto-load on import for library users / runners
load_project_env(quiet=True)
