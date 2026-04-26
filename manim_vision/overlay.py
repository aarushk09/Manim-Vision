"""Collision overlay replay utilities for diagnostic renders."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon as ShapelyPolygon


def serialize_overlap_geometry(geometry: Any) -> dict[str, Any]:
    """Serialize polygonal Shapely overlap geometry into world-space coordinates."""
    polygons: list[dict[str, Any]] = []
    if isinstance(geometry, ShapelyPolygon):
        polygons.append(_serialize_polygon(geometry))
    elif isinstance(geometry, MultiPolygon):
        polygons.extend(_serialize_polygon(poly) for poly in geometry.geoms)
    elif isinstance(geometry, GeometryCollection):
        for part in geometry.geoms:
            if isinstance(part, ShapelyPolygon):
                polygons.append(_serialize_polygon(part))
            elif isinstance(part, MultiPolygon):
                polygons.extend(_serialize_polygon(poly) for poly in part.geoms)
    return {"polygons": polygons}


def _serialize_polygon(polygon: ShapelyPolygon) -> dict[str, Any]:
    shell = [[float(x), float(y)] for x, y in polygon.exterior.coords]
    holes = [[[float(x), float(y)] for x, y in interior.coords] for interior in polygon.interiors]
    return {"shell": shell, "holes": holes}


@dataclass
class CollisionOverlayController:
    """Replay sampled overlap geometry as Manim overlay mobjects."""

    scene_name: str
    collision_events: list[dict[str, Any]]
    instant_visibility_seconds: float = 1 / 15

    @classmethod
    def from_report_path(cls, report_path: str | Path) -> "CollisionOverlayController":
        payload = json.loads(Path(report_path).read_text(encoding="utf-8"))
        return cls(
            scene_name=str(payload.get("scene", "UnknownScene")),
            collision_events=list(payload.get("collision_events") or []),
        )

    def active_samples_at_time(self, timestamp: float) -> list[dict[str, Any]]:
        """Return the active geometry sample for each collision interval at ``timestamp``."""
        active: list[dict[str, Any]] = []
        for event in self.collision_events:
            start_time = float(event.get("start_time", 0.0))
            end_time = float(event.get("end_time", 0.0))
            if end_time <= start_time:
                end_time = start_time + float(self.instant_visibility_seconds)
            if timestamp < start_time or timestamp > end_time:
                continue
            active.append(self._sample_for_time(event, timestamp))
        return active

    def overlay_group_at_time(self, timestamp: float) -> Any:
        """Build a Manim ``VGroup`` for every active collision at ``timestamp``."""
        from manim import Polygon, VGroup

        group = VGroup()
        group._manim_vision_ignore = True
        group.z_index = 10_000

        for sample in self.active_samples_at_time(timestamp):
            geometry = sample.get("geometry") or {}
            for polygon in geometry.get("polygons") or []:
                shell = polygon.get("shell") or []
                if len(shell) < 3:
                    continue
                points = [np.array([float(x), float(y), 0.0], dtype=np.float64) for x, y in shell]
                overlay = Polygon(
                    *points,
                    stroke_color="#00F5FF",
                    stroke_width=6,
                    fill_color="#FF2D55",
                    fill_opacity=0.85,
                )
                overlay._manim_vision_ignore = True
                overlay.z_index = 10_000
                group.add(overlay)
        return group

    def install(self, scene: Any) -> Any:
        """Attach a live-updating overlay layer to ``scene``."""
        from manim import VGroup

        frame_rate = getattr(getattr(scene, "camera", None), "frame_rate", None)
        if frame_rate:
            self.instant_visibility_seconds = 1 / float(frame_rate)

        layer = VGroup()
        layer._manim_vision_ignore = True
        layer.z_index = 10_000

        def updater(mob: Any) -> Any:
            current_time = float(getattr(getattr(scene, "renderer", None), "time", 0.0))
            mob.become(self.overlay_group_at_time(current_time))
            mob._manim_vision_ignore = True
            mob.z_index = 10_000
            return mob

        layer.add_updater(updater)
        add_foreground = getattr(scene, "add_foreground_mobject", None)
        if callable(add_foreground):
            add_foreground(layer)
        else:
            scene.add(layer)
        return layer

    def _sample_for_time(self, event: dict[str, Any], timestamp: float) -> dict[str, Any]:
        samples = list(event.get("samples") or [])
        if not samples:
            return {
                "time": float(event.get("start_time", 0.0)),
                "geometry": event.get("peak_geometry") or {"polygons": []},
            }
        chosen = samples[0]
        for sample in samples:
            if float(sample.get("time", 0.0)) <= timestamp:
                chosen = sample
            else:
                break
        return chosen


def overlay_mode_enabled() -> bool:
    """Whether the current render should attach collision overlays instead of rewriting logs."""
    return os.environ.get("MANIM_VISION_OVERLAY_MODE", "").lower() in {"1", "true", "yes"}


def overlay_report_path() -> Path | None:
    """Report path that should be replayed for overlay mode, if configured."""
    raw = os.environ.get("MANIM_VISION_OVERLAY_REPORT", "").strip()
    if not raw:
        return None
    return Path(raw)


def maybe_install_overlay(scene: Any) -> Any | None:
    """Attach a collision overlay layer to ``scene`` when overlay mode is enabled."""
    if not overlay_mode_enabled():
        return None
    report_path = overlay_report_path()
    if report_path is None or not report_path.exists():
        return None
    controller = CollisionOverlayController.from_report_path(report_path)
    return controller.install(scene)


def render_overlay_subprocess(
    *,
    script_path: Path,
    scene_name: str,
    report_path: Path,
    output_file: str,
    quality_flag: str = "-ql",
) -> Path:
    """Re-render ``scene_name`` with overlay mode enabled and a separate output filename."""
    env = os.environ.copy()
    env["MANIM_VISION_OVERLAY_MODE"] = "1"
    env["MANIM_VISION_OVERLAY_REPORT"] = str(report_path)
    env["MANIM_VISION_DISABLE_REPORT_WRITE"] = "1"
    cmd = [
        sys.executable,
        "-m",
        "manim",
        str(script_path),
        scene_name,
        quality_flag,
        "-o",
        output_file,
    ]
    subprocess.run(cmd, check=True, env=env, cwd=str(script_path.parent))
    return Path(output_file)
