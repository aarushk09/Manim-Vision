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
from manim_vision.telemetry.paths import check_digest_path_next_to_spatial_jsonl, default_report_paths
from manim_vision.telemetry.schema import MANIM_VISION_SPATIAL_REPORT_SCHEMA
from manim_vision.semantic import session_dedupe_enabled, stable_pair_key

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
    ) -> None:
        """Create a dispatcher.

        If ``output_stream`` is set (e.g. tests), all JSON goes there with pretty printing.

        Otherwise, reports are appended to ``jsonl_path`` and ``text_path`` (defaults:
        :func:`~manim_vision.telemetry.paths.default_report_paths`). Set
        ``MANIM_VISION_REPORT_STDOUT=1`` to also print each event to stdout.

        Args:
            output_stream: Override stream (bypasses file output).
            scene_name: Manim scene class name embedded in payloads and defaults.
            jsonl_path: One JSON object per line; created if needed.
            text_path: Human-readable log with separators.
            also_stdout: Print each event to ``sys.stdout`` as well.
        """
        self._default_scene_name = scene_name
        self._logger = logging.getLogger("manim_vision.telemetry")
        self._text_stream: TextIO | None = None
        self._digest_stream: TextIO | None = None
        self._digest_path: Path | None = None
        self._owns = False
        self._jsonl_path: Path | None = None
        self._txt_path: Path | None = None
        if output_stream is not None:
            self._json_stream: TextIO = output_stream
            self._session_dedupe = False
            self._dedupe_keys: set[str] = set()
            return

        d_json, d_txt = default_report_paths(scene_name)
        if jsonl_path is None:
            jsonl_path = d_json
        if text_path is None:
            text_path = d_txt
        self._jsonl_path = Path(jsonl_path)
        self._txt_path = Path(text_path)
        self._digest_path = (
            Path(check_digest_path) if check_digest_path is not None else check_digest_path_next_to_spatial_jsonl(self._jsonl_path)
        )
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

    def write_check_digest(self, report: dict[str, Any]) -> None:
        """Append one compact JSON line per play-check (merged pairs; default feedback channel).

        The digest is **not** validated by the per-event JSON Schema; it is a summary object.
        """
        if not self._owns or self._digest_path is None:
            return
        ts = (
            datetime.datetime.now(datetime.timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )
        body = {**report, "timestamp": report.get("timestamp", ts)}
        line = json.dumps(body, ensure_ascii=False, separators=(",", ":")) + "\n"
        if self._digest_stream is None:
            self._digest_path.parent.mkdir(parents=True, exist_ok=True)
            self._digest_stream = self._digest_path.open("a", encoding="utf-8")
        self._digest_stream.write(line)
        self._digest_stream.flush()
        if self._text_stream is not None:
            am = body.get("actionable_merged") or []
            sup = body.get("suppressed") or {}
            n_sup = 0
            if isinstance(sup, dict) and all(isinstance(x, (int, float)) for x in sup.values()):
                n_sup = int(sum(sup.values()))
            self._text_stream.write(
                f"\n[manim-vision digest]  ts={ts}  actionable={len(am)}  suppressed_hits~={n_sup}  "
                f"raw_pair_hits={body.get('raw_pair_hits', '?')}\n"
            )
            self._text_stream.flush()

    def close(self) -> None:
        """Close opened report files. Safe to call more than once."""
        if not self._owns:
            return
        for s in (getattr(self, "_json_stream", None), self._text_stream, self._digest_stream):
            if s and not s.closed and s is not sys.stdout and s is not sys.stderr:
                try:
                    s.flush()
                finally:
                    s.close()
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
        """Build, validate, and dispatch a Spatial Health Report.

        When ``self._session_dedupe`` is active (default for file output), a pair
        of ``entity_labels`` that was already written in this render is skipped
        and ``None`` is returned.

        Args:
            collision_result: A :class:`~manim_vision.geometry.engine.CollisionResult` instance.
            mtv: Iterable MTV components (length at least two).
            fix_syntax: Manim ``shift`` chain or comment string.
            scene_name: Optional override for the reporting scene name.
            entity_labels: Human-scale ``(a, b)`` strings for ``colliding_entities``; default is raw mobject names.

        Returns:
            The validated payload dictionary, or ``None`` if de-duplicated.

        Raises:
            ManimVisionSchemaError: If the payload fails JSON Schema validation.
        """
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
            "timestamp": datetime.datetime.now(datetime.timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
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
            line = json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
            )
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
