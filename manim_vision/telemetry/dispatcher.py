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

_TEXT_LABEL_RE = re.compile(r'^Text\("(?P<text>.*)"\)(?:\[\d+\])?$')


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


def _extract_text(label: str) -> str | None:
    match = _TEXT_LABEL_RE.fullmatch(label)
    if not match:
        return None
    return match.group("text")


def _is_numeric_text_label(label: str) -> bool:
    text = _extract_text(label)
    return bool(text and text.isdigit())


def _is_story_text_label(label: str) -> bool:
    text = _extract_text(label)
    return bool(text and not text.isdigit() and text not in {"", "Find14"})


def _is_array_element(label: str) -> bool:
    return label.startswith("Square") or _is_numeric_text_label(label)


def _is_search_ui_label(label: str) -> bool:
    text = _extract_text(label)
    return _is_array_element(label) or text == "Find14"


def _is_arrow_label(label: str) -> bool:
    return label.startswith("Arrow") or label.startswith("ArrowTriangle")


def _compact_fix(fix: str) -> str:
    return fix.replace("shift(", "").replace(")", "")


def _issue_key(pair: tuple[str, str]) -> str:
    a, b = pair
    labels = (a, b)
    if all(_is_story_text_label(label) for label in labels):
        return "title_spacing"
    if any(_is_arrow_label(label) for label in labels) and any(_is_story_text_label(label) for label in labels):
        return "marker_vs_text"
    if any(_is_search_ui_label(label) or label == "VMobjectFromSVGPath" for label in labels) and any(
        _is_story_text_label(label) for label in labels
    ):
        return "stale_search_ui"
    return stable_pair_key(a, b)


def _contains_generic_fragment(pair: tuple[str, str]) -> bool:
    return any(label in {"VMobjectFromSVGPath", "VectorizedPoint"} for label in pair)


def _issue_sort_key(issue_key: str) -> tuple[int, str]:
    order = {
        "title_spacing": 0,
        "marker_vs_text": 1,
        "stale_search_ui": 2,
    }
    return (order.get(issue_key, 99), issue_key)


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
        self._summary_reports: list[dict[str, Any]] = []
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
        if self._results is None and self._summary_reports:
            self._results = self._build_scene_summary()
        return self._results

    def write_check_digest(self, report: dict[str, Any]) -> None:
        """Record per-check event batches or append them immediately in legacy mode."""
        if self._output_mode in {"llm", "human", "silent"}:
            ts = report.get("timestamp", _utc_now())
            body = {**report, "timestamp": ts}
            self._summary_reports.append(body)
            self._results = None
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

    def _build_scene_summary(self) -> dict[str, Any]:
        issue_buckets: dict[str, dict[str, Any]] = {}
        total_events = 0
        for report in self._summary_reports:
            for item in report.get("actionable_merged") or []:
                pair = tuple(item.get("pair") or ())
                if len(pair) != 2:
                    continue
                total_events += 1
                key = _issue_key(pair)
                bucket = issue_buckets.setdefault(
                    key,
                    {
                        "key": key,
                        "pairs": [],
                        "targets": set(),
                        "sources": set(),
                        "samples": [],
                    },
                )
                bucket["pairs"].append(pair)
                texts = [text for text in (_extract_text(pair[0]), _extract_text(pair[1])) if text]
                for text in texts:
                    if text and not text.isdigit():
                        bucket["targets"].add(text)
                for label in pair:
                    bucket["sources"].add(label)
                bucket["samples"].append(item)

        issues: list[dict[str, Any]] = []
        for key in sorted(issue_buckets, key=_issue_sort_key):
            bucket = issue_buckets[key]
            sample = max(bucket["samples"], key=lambda item: float(item.get("max_overlap_area", 0.0)))
            if key == "title_spacing":
                texts = sorted(bucket["targets"])[:2]
                issues.append(
                    {
                        "problem": "title/subtitle overlap",
                        "objects": texts or list(sample["pair"]),
                        "fix": sample["fix_suggestion"],
                    }
                )
                continue
            if key == "marker_vs_text":
                targets = [text for text in sorted(bucket["targets"]) if text]
                issues.append(
                    {
                        "problem": "pointer tip overlaps intro text",
                        "objects": targets[:1] or list(sample["pair"]),
                        "fix": sample["fix_suggestion"],
                    }
                )
                continue
            if key == "stale_search_ui":
                targets = [text for text in sorted(bucket["targets"]) if text and text != "Find14"]
                issues.append(
                    {
                        "problem": "old search UI overlaps later sections",
                        "objects": targets[:4],
                        "fix": "fade out the array row, index labels, arrows, and the Find 14 label before the later narration, requirement, and ending sections",
                    }
                )
                continue

            if _contains_generic_fragment(sample["pair"]):
                continue
            issues.append(
                {
                    "problem": "overlap",
                    "objects": list(sample["pair"]),
                    "fix": sample["fix_suggestion"],
                }
            )

        return {
            "scene": self._default_scene_name,
            "events": total_events,
            "issues": issues,
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

        lines = [f"scene: {summary['scene']}", f"issues: {len(summary['issues'])}"]
        for issue in summary.get("issues", []):
            objects = ", ".join(issue.get("objects") or [])
            lines.append(f"- {issue['problem']}: {objects}")
            lines.append(f"  fix: {issue['fix']}")
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
