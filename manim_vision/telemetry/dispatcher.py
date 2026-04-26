"""Build, validate, and emit JSON spatial health reports."""

from __future__ import annotations

import datetime
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, TextIO

from jsonschema import ValidationError, validate

from manim_vision.exceptions import ManimVisionSchemaError
from manim_vision.semantic import session_dedupe_enabled, stable_pair_key
from manim_vision.telemetry.paths import check_digest_path_next_to_spatial_jsonl, default_report_paths
from manim_vision.telemetry.schema import MANIM_VISION_SPATIAL_REPORT_SCHEMA

logger = logging.getLogger("manim_vision.telemetry")


def _text_block(payload: dict[str, Any], scene_label: str) -> str:
    mtv = payload.get("resolution_mtv") or {}
    return (
        f"\n{'=' * 72}\n"
        f"[{payload.get('timestamp', '')}]  scene: {scene_label}\n"
        f"{payload.get('error_type', '')}  overlap_area={payload.get('overlap_area')}\n"
        f"entities:  {', '.join(payload.get('colliding_entities') or [])}\n"
        f"mtv:  x={mtv.get('x')}  y={mtv.get('y')}  z={mtv.get('z')}\n"
        f"fix:  {payload.get('fix_suggestion', '')}\n"
        f"{'=' * 72}\n"
    )


def _utc_now() -> str:
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _round_float(value: Any, digits: int = 4) -> float:
    """Return a stable float for JSON output and text logs."""
    return round(float(value), digits)


def _format_event_text(event: dict[str, Any]) -> str:
    """Render one collision interval for the human-readable report."""
    objects = " <-> ".join(event.get("objects") or [])
    centroid = event.get("peak_centroid") or {}
    mtv = event.get("resolution_mtv") or {}
    return (
        f"- {objects}\n"
        f"  interval: {event.get('start_time', 0.0):.2f}s -> {event.get('end_time', 0.0):.2f}s"
        f" ({event.get('duration', 0.0):.2f}s)\n"
        f"  peak overlap: area={event.get('peak_overlap_area', 0.0):.4f}"
        f" centroid=({centroid.get('x', 0.0):.4f}, {centroid.get('y', 0.0):.4f})\n"
        f"  mtv: x={mtv.get('x', 0.0):.4f} y={mtv.get('y', 0.0):.4f} z={mtv.get('z', 0.0):.4f}\n"
        f"  fix: {event.get('fix_suggestion', '')}"
    )


class TelemetryDispatcher:
    """Serializes collision telemetry to files (and optional stdout) after schema validation."""

    def __init__(
        self,
        output_stream: TextIO | None = None,
        scene_name: str = "UnknownScene",
        *,
        jsonl_path: Path | str | None = None,
        text_path: Path | str | None = None,
        check_digest_path: Path | str | None = None,
        also_stdout: bool | None = None,
        output_mode: str = "legacy",
    ) -> None:
        """Create a dispatcher.

        Modes:
        - ``legacy``: write every payload immediately (current low-level behavior).
        - ``llm``: collect new events and flush one compact scene summary on close.
        - ``human``: collect events and flush a readable scene summary on close.
        - ``silent``: collect events for programmatic access only.
        """
        self._default_scene_name = scene_name
        self._logger = logging.getLogger("manim_vision.telemetry")
        self._text_stream: TextIO | None = None
        self._digest_stream: TextIO | None = None
        self._digest_path: Path | None = None
        self._owns = False
        self._jsonl_path: Path | None = None
        self._txt_path: Path | None = None
        self._output_mode = output_mode
        self._collision_events: list[dict[str, Any]] = []
        self._results: dict[str, Any] | None = None
        if output_stream is not None:
            self._json_stream: TextIO = output_stream
            self._session_dedupe = False
            self._dedupe_keys: set[str] = set()
            self._also_stdout = False
            return

        d_json, d_txt = default_report_paths(scene_name)
        if jsonl_path is None:
            jsonl_path = d_json
        if text_path is None:
            text_path = d_txt
        self._jsonl_path = Path(jsonl_path)
        self._txt_path = Path(text_path)
        self._digest_path = (
            Path(check_digest_path)
            if check_digest_path is not None
            else check_digest_path_next_to_spatial_jsonl(self._jsonl_path)
        )
        if self._output_mode == "silent":
            self._json_stream = sys.stdout
            self._session_dedupe = False
            self._dedupe_keys = set()
            self._also_stdout = False
            return
        self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self._txt_path.parent.mkdir(parents=True, exist_ok=True)
        self._json_stream = self._jsonl_path.open("a", encoding="utf-8")
        self._text_stream = self._txt_path.open("a", encoding="utf-8")
        self._owns = True
        if also_stdout is not None:
            self._also_stdout = also_stdout
        else:
            from os import environ

            self._also_stdout = environ.get("MANIM_VISION_REPORT_STDOUT", "").lower() in (
                "1",
                "true",
                "yes",
            )
        self._session_dedupe = self._owns and session_dedupe_enabled()
        self._dedupe_keys: set[str] = set()

    @property
    def report_jsonl_path(self) -> Path | None:
        """File receiving JSONL when using default file output; ``None`` if a stream was passed."""
        return self._jsonl_path if self._owns else None

    @property
    def report_text_path(self) -> Path | None:
        """Human-readable log file path, or ``None`` if a stream was passed."""
        return self._txt_path if self._owns else None

    @property
    def check_digest_path(self) -> Path | None:
        """``*_check_digest.jsonl`` (lazy-created on first :meth:`write_check_digest`)."""
        return self._digest_path if self._owns else None

    @property
    def results(self) -> dict[str, Any] | None:
        """Return the built scene summary after ``close()``, or build it lazily if needed."""
        if self._results is None and self._collision_events:
            self._results = self._build_scene_summary()
        return self._results

    def write_check_digest(self, report: dict[str, Any]) -> None:
        """Record per-check event batches or append them immediately in legacy mode."""
        if self._output_mode in {"llm", "human", "silent"}:
            for event in report.get("collision_events") or []:
                self.record_collision_event(event)
            return

        if not self._owns or self._digest_path is None:
            return
        ts = report.get("timestamp", _utc_now())
        body = {**report, "timestamp": ts}
        line = json.dumps(body, ensure_ascii=False, separators=(",", ":")) + "\n"
        if self._digest_stream is None:
            self._digest_path.parent.mkdir(parents=True, exist_ok=True)
            self._digest_stream = self._digest_path.open("a", encoding="utf-8")
        self._digest_stream.write(line)
        self._digest_stream.flush()
        if self._text_stream is not None:
            actionable = body.get("actionable_merged") or []
            suppressed = body.get("suppressed") or {}
            suppressed_count = 0
            if isinstance(suppressed, dict) and all(isinstance(x, (int, float)) for x in suppressed.values()):
                suppressed_count = int(sum(suppressed.values()))
            self._text_stream.write(
                f"\n[manim-vision digest]  ts={ts}  actionable={len(actionable)}  suppressed_hits~={suppressed_count}  "
                f"raw_pair_hits={body.get('raw_pair_hits', '?')}\n"
            )
            self._text_stream.flush()

    def record_collision_event(self, event: dict[str, Any]) -> None:
        """Append one finalized collision interval for later output."""
        centroid = event.get("peak_centroid") or {}
        mtv = event.get("resolution_mtv") or {}
        body = {
            "objects": list(event.get("objects") or []),
            "start_time": _round_float(event.get("start_time", 0.0), 3),
            "end_time": _round_float(event.get("end_time", 0.0), 3),
            "duration": _round_float(event.get("duration", 0.0), 3),
            "peak_overlap_area": _round_float(event.get("peak_overlap_area", 0.0), 4),
            "peak_centroid": {
                "x": _round_float(centroid.get("x", 0.0), 4),
                "y": _round_float(centroid.get("y", 0.0), 4),
            },
            "resolution_mtv": {
                "x": _round_float(mtv.get("x", 0.0), 4),
                "y": _round_float(mtv.get("y", 0.0), 4),
                "z": _round_float(mtv.get("z", 0.0), 4),
            },
            "fix_suggestion": str(event.get("fix_suggestion", "")),
        }
        self._collision_events.append(body)
        self._results = None

    def _build_scene_summary(self) -> dict[str, Any]:
        events = sorted(
            self._collision_events,
            key=lambda event: (
                float(event.get("start_time", 0.0)),
                float(event.get("end_time", 0.0)),
                tuple(event.get("objects") or ()),
            ),
        )
        return {
            "scene": self._default_scene_name,
            "collision_events": events,
        }

    def _write_summary_outputs(self, summary: dict[str, Any]) -> None:
        if self._output_mode == "silent":
            return

        if self._output_mode == "llm":
            payload = json.dumps(summary, ensure_ascii=False, separators=(",", ":"))
            if self._owns and self._digest_path is not None:
                if self._digest_stream is None:
                    self._digest_path.parent.mkdir(parents=True, exist_ok=True)
                    self._digest_stream = self._digest_path.open("w", encoding="utf-8")
                self._digest_stream.write(payload + "\n")
                self._digest_stream.flush()
            else:
                self._json_stream.write(payload + "\n")
                self._json_stream.flush()
            return

        events = summary.get("collision_events") or []
        lines = [f"scene: {summary['scene']}", f"collision_events: {len(events)}"]
        for event in events:
            lines.append(_format_event_text(event))
        body = "\n".join(lines) + "\n"
        if self._owns and self._text_stream is not None:
            self._text_stream.write(body)
            self._text_stream.flush()
        else:
            self._json_stream.write(body)
            self._json_stream.flush()

    def close(self) -> None:
        """Flush summary output for summary modes, then close opened report files."""
        if self._output_mode in {"llm", "human", "silent"}:
            self._results = self._build_scene_summary()
            self._write_summary_outputs(self._results)

        if not self._owns:
            return
        for stream in (getattr(self, "_json_stream", None), self._text_stream, self._digest_stream):
            if stream and not stream.closed and stream is not sys.stdout and stream is not sys.stderr:
                try:
                    stream.flush()
                finally:
                    stream.close()
        self._digest_stream = None
        self._owns = False

    def dispatch(
        self,
        collision_result: Any,
        mtv: Any,
        fix_syntax: str,
        *,
        scene_name: str | None = None,
        entity_labels: tuple[str, str] | None = None,
    ) -> dict[str, Any] | None:
        """Build, validate, and dispatch a Spatial Health Report."""
        name = scene_name or self._default_scene_name
        if isinstance(name, str):
            name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
        if not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z0-9_]+", name):
            self._logger.warning("Sanitizing scene name for schema pattern compliance.")
            name = "UnknownScene"
        display_name = name

        ent_a, ent_b = (
            (entity_labels[0], entity_labels[1])
            if entity_labels is not None
            else (collision_result.mobject_a_name, collision_result.mobject_b_name)
        )
        pair_key = stable_pair_key(ent_a, ent_b)
        if self._session_dedupe and pair_key in self._dedupe_keys:
            return None

        payload = {
            "timestamp": _utc_now(),
            "scene_name": name,
            "error_type": "OVERLAP",
            "colliding_entities": [ent_a, ent_b],
            "overlap_area": float(collision_result.overlap_area),
            "resolution_mtv": {
                "x": float(mtv[0]),
                "y": float(mtv[1]),
                "z": float(mtv[2]) if len(mtv) > 2 else 0.0,
            },
            "fix_suggestion": fix_syntax,
        }

        try:
            validate(instance=payload, schema=MANIM_VISION_SPATIAL_REPORT_SCHEMA)
        except ValidationError as exc:
            raise ManimVisionSchemaError(
                f"Generated telemetry payload failed schema validation: {exc.message}"
            ) from exc
        if self._session_dedupe:
            self._dedupe_keys.add(pair_key)

        if self._owns:
            line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            self._json_stream.write(line + "\n")
            self._json_stream.flush()
            if self._text_stream is not None:
                self._text_stream.write(_text_block(payload, display_name))
                self._text_stream.flush()
        else:
            serialized = json.dumps(payload, indent=2)
            self._json_stream.write(serialized + "\n")
            self._json_stream.flush()

        if self._owns and self._also_stdout:
            print(json.dumps(payload, indent=2, ensure_ascii=False), file=sys.stdout, flush=True)

        return payload
