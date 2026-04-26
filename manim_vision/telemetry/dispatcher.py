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
_COMPONENT_SUFFIX_RE = re.compile(r"^(?P<owner>.+?)\.(?P<role>char|glyph|part)\[(?P<index>\d+)\]$")


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


def _event_sort_key(event: dict[str, Any]) -> tuple[float, float, tuple[str, ...]]:
    return (
        float(event.get("start_time", 0.0)),
        float(event.get("end_time", 0.0)),
        tuple(event.get("objects") or ()),
    )


def _partner_for_anchor(event: dict[str, Any], anchor: str) -> str | None:
    objects = list(event.get("objects") or [])
    if len(objects) != 2 or anchor not in objects:
        return None
    return objects[1] if objects[0] == anchor else objects[0]


def _compress_indices(indices: list[int]) -> str:
    if not indices:
        return ""
    ranges: list[str] = []
    start = prev = indices[0]
    for index in indices[1:]:
        if index == prev + 1:
            prev = index
            continue
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = index
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(ranges)


def _summarize_component_side(owner: str, labels: list[str]) -> str:
    """Collapse many leaf labels into one owner-relative component summary."""
    if not labels:
        return "none"
    roles: dict[str, list[int]] = {}
    extras: list[str] = []
    whole_object = False
    for label in sorted(set(labels)):
        if label == owner:
            whole_object = True
            continue
        match = _COMPONENT_SUFFIX_RE.match(label)
        if match and match.group("owner") == owner:
            roles.setdefault(match.group("role"), []).append(int(match.group("index")))
            continue
        if label.startswith(owner + "."):
            extras.append(label[len(owner) + 1 :])
        else:
            extras.append(label)

    if whole_object and not roles and not extras:
        return "whole"

    parts: list[str] = []
    for role in sorted(roles):
        parts.append(f"{role}[{_compress_indices(sorted(roles[role]))}]")
    parts.extend(sorted(set(extras)))
    if whole_object:
        parts.insert(0, "whole")
    return ", ".join(parts) if parts else "whole"


def _public_hotspot(raw_hotspot: dict[str, Any]) -> dict[str, Any]:
    centroid = raw_hotspot.get("centroid") or {}
    return {
        "centroid": {
            "x": _round_float(centroid.get("x", 0.0), 4),
            "y": _round_float(centroid.get("y", 0.0), 4),
        },
        "area": _round_float(raw_hotspot.get("area", 0.0), 4),
        "contact": list(raw_hotspot.get("contact") or ["", ""]),
    }


def _public_event(raw_event: dict[str, Any]) -> dict[str, Any]:
    objects = list(raw_event.get("objects") or [])
    centroid = raw_event.get("peak_centroid") or {}
    mtv = raw_event.get("resolution_mtv") or {}
    raw_components = list(raw_event.get("components") or [[], []])
    left_components = list(raw_components[0] if len(raw_components) > 0 else [])
    right_components = list(raw_components[1] if len(raw_components) > 1 else [])
    return {
        "objects": objects,
        "start_time": _round_float(raw_event.get("start_time", 0.0), 3),
        "end_time": _round_float(raw_event.get("end_time", 0.0), 3),
        "duration": _round_float(raw_event.get("duration", 0.0), 3),
        "peak_overlap_area": _round_float(raw_event.get("peak_overlap_area", 0.0), 4),
        "peak_centroid": {
            "x": _round_float(centroid.get("x", 0.0), 4),
            "y": _round_float(centroid.get("y", 0.0), 4),
        },
        "contact_summary": [
            _summarize_component_side(objects[0], left_components) if len(objects) > 0 else "none",
            _summarize_component_side(objects[1], right_components) if len(objects) > 1 else "none",
        ],
        "hotspots": [_public_hotspot(hotspot) for hotspot in list(raw_event.get("hotspots") or [])[:3]],
        "resolution_mtv": {
            "x": _round_float(mtv.get("x", 0.0), 4),
            "y": _round_float(mtv.get("y", 0.0), 4),
            "z": _round_float(mtv.get("z", 0.0), 4),
        },
        "fix_suggestion": str(raw_event.get("fix_suggestion", "")),
    }


def _overlay_event(raw_event: dict[str, Any]) -> dict[str, Any]:
    return {
        "objects": list(raw_event.get("objects") or []),
        "start_time": _round_float(raw_event.get("start_time", 0.0), 3),
        "end_time": _round_float(raw_event.get("end_time", 0.0), 3),
        "peak_geometry": raw_event.get("peak_geometry") or {"polygons": []},
        "samples": list(raw_event.get("samples") or []),
    }


def _format_event_text(event: dict[str, Any]) -> str:
    """Render one collision interval for the human-readable report."""
    objects = " <-> ".join(event.get("objects") or [])
    centroid = event.get("peak_centroid") or {}
    mtv = event.get("resolution_mtv") or {}
    contacts = list(event.get("contact_summary") or ["", ""])
    hotspots = list(event.get("hotspots") or [])
    hotspot_bits = []
    for hotspot in hotspots:
        point = hotspot.get("centroid") or {}
        hotspot_bits.append(
            f"({point.get('x', 0.0):.3f},{point.get('y', 0.0):.3f}) a={hotspot.get('area', 0.0):.4f}"
        )
    hotspot_text = "; ".join(hotspot_bits) if hotspot_bits else "none"
    return (
        f"- {objects}\n"
        f"  interval: {event.get('start_time', 0.0):.2f}s -> {event.get('end_time', 0.0):.2f}s"
        f" ({event.get('duration', 0.0):.2f}s)\n"
        f"  peak overlap: area={event.get('peak_overlap_area', 0.0):.4f}"
        f" centroid=({centroid.get('x', 0.0):.4f}, {centroid.get('y', 0.0):.4f})\n"
        f"  contact: {contacts[0]}  x  {contacts[1]}\n"
        f"  hotspots: {hotspot_text}\n"
        f"  mtv: x={mtv.get('x', 0.0):.4f} y={mtv.get('y', 0.0):.4f} z={mtv.get('z', 0.0):.4f}\n"
        f"  fix: {event.get('fix_suggestion', '')}"
    )


def _build_collision_groups(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed_events = list(enumerate(events))
    anchor_to_events: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for index, event in indexed_events:
        for obj in list(event.get("objects") or []):
            anchor_to_events.setdefault(obj, []).append((index, event))

    candidates: list[tuple[str, int, int, float]] = []
    for anchor, anchor_events in anchor_to_events.items():
        partners = {
            partner
            for _index, event in anchor_events
            if (partner := _partner_for_anchor(event, anchor)) is not None
        }
        if len(partners) < 2:
            continue
        first_seen = min(float(event.get("start_time", 0.0)) for _index, event in anchor_events)
        candidates.append((anchor, len(partners), len(anchor_events), first_seen))

    candidates.sort(key=lambda item: (-item[1], -item[2], item[3], item[0]))

    unassigned = {index for index, _event in indexed_events}
    groups: list[dict[str, Any]] = []

    for anchor, _partner_count, _event_count, _first_seen in candidates:
        available = [(index, event) for index, event in anchor_to_events[anchor] if index in unassigned]
        partner_map: dict[str, list[dict[str, Any]]] = {}
        for _index, event in available:
            partner = _partner_for_anchor(event, anchor)
            if partner is None:
                continue
            partner_map.setdefault(partner, []).append(event)
        if len(partner_map) < 2:
            continue

        members = []
        consumed: set[int] = set()
        for partner, partner_events in sorted(
            partner_map.items(),
            key=lambda item: (_event_sort_key(item[1][0]), item[0]),
        ):
            ordered_partner_events = sorted(partner_events, key=_event_sort_key)
            members.append({"object": partner, "events": ordered_partner_events})
            consumed.update(
                index
                for index, event in available
                if _partner_for_anchor(event, anchor) == partner
            )

        groups.append(
            {
                "kind": "anchor",
                "anchor": anchor,
                "objects": [anchor, *[member["object"] for member in members]],
                "partner_count": len(members),
                "event_count": sum(len(member["events"]) for member in members),
                "members": members,
            }
        )
        unassigned.difference_update(consumed)

    pair_map: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for index in sorted(unassigned, key=lambda item: _event_sort_key(events[item])):
        event = events[index]
        pair_map.setdefault(tuple(sorted(event.get("objects") or [])), []).append(event)

    for pair, pair_events in pair_map.items():
        groups.append(
            {
                "kind": "pair",
                "objects": list(pair),
                "event_count": len(pair_events),
                "events": pair_events,
            }
        )
    return groups


def _format_group_text(group: dict[str, Any]) -> str:
    if group.get("kind") == "anchor":
        lines = [
            f"* anchor: {group.get('anchor', '')}",
            f"  partners: {group.get('partner_count', 0)}  events: {group.get('event_count', 0)}",
        ]
        for member in group.get("members") or []:
            lines.append(f"  with {member.get('object', '')}")
            for event in member.get("events") or []:
                lines.append("    " + _format_event_text(event).replace("\n", "\n    "))
        return "\n".join(lines)

    lines = []
    for event in group.get("events") or []:
        lines.append(_format_event_text(event))
    return "\n".join(lines)


def _build_final_context_report(summary: dict[str, Any]) -> dict[str, Any]:
    events = list(summary.get("collision_events") or [])
    groups = list(summary.get("collision_groups") or [])

    object_labels: list[str] = []
    object_index: dict[str, int] = {}

    def intern_object(label: str) -> int:
        if label not in object_index:
            object_index[label] = len(object_labels)
            object_labels.append(label)
        return object_index[label]

    event_rows: list[list[Any]] = []
    for event in events:
        objects = list(event.get("objects") or [])
        centroid = event.get("peak_centroid") or {}
        contacts = list(event.get("contact_summary") or ["", ""])
        hotspots = []
        for hotspot in list(event.get("hotspots") or [])[:1]:
            point = hotspot.get("centroid") or {}
            contact = list(hotspot.get("contact") or ["", ""])
            hotspots.append(
                [
                    point.get("x", 0.0),
                    point.get("y", 0.0),
                    hotspot.get("area", 0.0),
                    contact[0],
                    contact[1],
                ]
            )
        event_rows.append(
            [
                intern_object(objects[0]),
                intern_object(objects[1]),
                event.get("start_time", 0.0),
                event.get("end_time", 0.0),
                event.get("peak_overlap_area", 0.0),
                centroid.get("x", 0.0),
                centroid.get("y", 0.0),
                contacts[0],
                contacts[1],
                str(event.get("fix_suggestion", "")),
                hotspots,
            ]
        )

    anchor_rows: list[list[Any]] = []
    event_ids_by_object: dict[str, list[int]] = {}
    for event_index, event in enumerate(events):
        for obj in list(event.get("objects") or []):
            event_ids_by_object.setdefault(obj, []).append(event_index)

    for group in groups:
        if group.get("kind") != "anchor":
            continue
        anchor = str(group.get("anchor", ""))
        anchor_rows.append([intern_object(anchor), event_ids_by_object.get(anchor, [])])

    return {
        "fmt": "v2 objs[i]=object; ev=[a,b,t0,t1,area,x,y,partsA,partsB,fix,h]; h=[x,y,area,partA,partB]; grp=[anchor,[ev]]",
        "scene": summary.get("scene", "UnknownScene"),
        "objs": object_labels,
        "ev": event_rows,
        "grp": anchor_rows,
    }


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
        - ``legacy``: write every payload immediately.
        - ``llm``: collect events and flush one semantic scene summary on close.
        - ``human``: collect events and flush a readable grouped summary on close.
        - ``silent``: collect events for programmatic access only.
        """
        self._default_scene_name = scene_name
        self._logger = logging.getLogger("manim_vision.telemetry")
        self._text_stream: TextIO | None = None
        self._digest_stream: TextIO | None = None
        self._digest_path: Path | None = None
        self._final_context_path: Path | None = None
        self._overlay_data_path: Path | None = None
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
        self._final_context_path = self._jsonl_path.with_name(
            f"{scene_name}_finalcontextcollisionreport.json"
        )
        self._overlay_data_path = self._jsonl_path.with_name(f"{scene_name}_overlaydata.json")
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
    def final_context_path(self) -> Path | None:
        """Compressed LLM-facing report path when using default file output."""
        return self._final_context_path

    @property
    def overlay_data_path(self) -> Path | None:
        """Geometry-rich collision replay payload for overlay rendering."""
        return self._overlay_data_path

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
            "peak_geometry": event.get("peak_geometry") or {"polygons": []},
            "samples": [
                {
                    "time": _round_float(sample.get("time", 0.0), 3),
                    "area": _round_float(sample.get("area", 0.0), 4),
                    "centroid": {
                        "x": _round_float((sample.get("centroid") or {}).get("x", 0.0), 4),
                        "y": _round_float((sample.get("centroid") or {}).get("y", 0.0), 4),
                    },
                    "geometry": sample.get("geometry") or {"polygons": []},
                }
                for sample in list(event.get("samples") or [])
            ],
            "components": [
                sorted(raw_component for raw_component in list((event.get("components") or [[], []])[0])),
                sorted(raw_component for raw_component in list((event.get("components") or [[], []])[1])),
            ],
            "hotspots": [
                {
                    "centroid": {
                        "x": _round_float((hotspot.get("centroid") or {}).get("x", 0.0), 4),
                        "y": _round_float((hotspot.get("centroid") or {}).get("y", 0.0), 4),
                    },
                    "area": _round_float(hotspot.get("area", 0.0), 4),
                    "contact": list(hotspot.get("contact") or ["", ""]),
                }
                for hotspot in list(event.get("hotspots") or [])[:3]
            ],
        }
        self._collision_events.append(body)
        self._results = None

    def _build_scene_summary(self) -> dict[str, Any]:
        public_events = sorted((_public_event(event) for event in self._collision_events), key=_event_sort_key)
        groups = _build_collision_groups(public_events)
        return {
            "version": "2.0",
            "scene": self._default_scene_name,
            "event_count": len(public_events),
            "group_count": len(groups),
            "collision_events": public_events,
            "collision_groups": groups,
        }

    def _build_overlay_payload(self) -> dict[str, Any]:
        overlay_events = sorted((_overlay_event(event) for event in self._collision_events), key=_event_sort_key)
        return {
            "scene": self._default_scene_name,
            "collision_events": overlay_events,
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
                if self._final_context_path is not None:
                    compact = _build_final_context_report(summary)
                    self._final_context_path.write_text(
                        json.dumps(compact, ensure_ascii=False, separators=(",", ":")) + "\n",
                        encoding="utf-8",
                    )
                if self._overlay_data_path is not None:
                    overlay_payload = self._build_overlay_payload()
                    self._overlay_data_path.write_text(
                        json.dumps(overlay_payload, ensure_ascii=False, separators=(",", ":")) + "\n",
                        encoding="utf-8",
                    )
            else:
                self._json_stream.write(payload + "\n")
                self._json_stream.flush()
            return

        events = summary.get("collision_events") or []
        groups = summary.get("collision_groups") or []
        lines = [
            f"scene: {summary['scene']}",
            f"collision_events: {len(events)}",
            f"collision_groups: {len(groups)}",
        ]
        for group in groups:
            lines.append(_format_group_text(group))
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
