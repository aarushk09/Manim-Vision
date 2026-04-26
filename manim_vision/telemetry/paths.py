"""Default paths for spatial health reports (under Manim’s ``media`` directory)."""

from __future__ import annotations

import os
import re
from pathlib import Path

_SAFE_SCENE = re.compile(r"[^a-zA-Z0-9_]+")


def _safe_scene_name(name: str) -> str:
    s = _SAFE_SCENE.sub("_", name.strip()) or "scene"
    return s[: 200]


def default_report_dir() -> Path:
    """Directory for Manim Vision report files; created if it does not exist.

    Precedence: ``MANIM_VISION_REPORT_DIR`` env, then ``{manim config media_dir}/manim_vision``,
    else ``./media/manim_vision`` relative to the process cwd.
    """
    override = os.environ.get("MANIM_VISION_REPORT_DIR", "").strip()
    if override:
        p = Path(override).expanduser()
    else:
        try:
            from manim import config as mcfg  # type: ignore[import-untyped]

            media = Path(str(getattr(mcfg, "media_dir", "media"))).expanduser()
        except Exception:
            media = Path("media")
        p = (media if media.is_absolute() else Path.cwd() / media) / "manim_vision"
    p.mkdir(parents=True, exist_ok=True)
    return p


def default_report_paths(scene_name: str) -> tuple[Path, Path]:
    """Return (jsonl_path, human_readable_txt_path) for a scene class name."""
    base = _safe_scene_name(scene_name)
    d = default_report_dir()
    return d / f"{base}_spatial.jsonl", d / f"{base}_spatial_log.txt"
