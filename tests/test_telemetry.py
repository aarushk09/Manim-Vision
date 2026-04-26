"""Tests for :mod:`manim_vision.telemetry.dispatcher`."""

from __future__ import annotations

import io
import json
from types import SimpleNamespace

import numpy as np
import pytest
from shapely.geometry import Polygon

from manim_vision.exceptions import ManimVisionSchemaError
from manim_vision.geometry.engine import CollisionResult
from manim_vision.telemetry.dispatcher import TelemetryDispatcher


def _sample_collision() -> CollisionResult:
    """Construct a minimal valid collision for telemetry."""
    a = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    b = Polygon([(0.5, 0), (1.5, 0), (1.5, 1), (0.5, 1)])
    inter = a.intersection(b)
    return CollisionResult(
        mobject_a_id=10,
        mobject_b_id=20,
        mobject_a_name="Circle_10",
        mobject_b_name="Circle_20",
        geom_a=a,
        geom_b=b,
        overlap_area=float(inter.area),
        overlap_centroid=tuple(inter.centroid.coords[0]),
        overlap_geometry=inter,
    )


def test_valid_payload_dispatched() -> None:
    """A valid collision and MTV must serialize to the output stream."""
    buf = io.StringIO()
    dispatcher = TelemetryDispatcher(output_stream=buf, scene_name="TestScene")
    payload = dispatcher.dispatch(_sample_collision(), np.array([0.1, -0.2, 0.0]), "shift(UP * 1.0)")
    assert payload["scene_name"] == "TestScene"
    assert len(buf.getvalue()) > 0


def test_schema_violation_raises_error() -> None:
    """Invalid overlap area must raise before anything is written to the stream."""
    buf = io.StringIO()
    dispatcher = TelemetryDispatcher(output_stream=buf, scene_name="BadScene")
    bad = SimpleNamespace(
        mobject_a_name="A_1",
        mobject_b_name="B_2",
        overlap_area=-1.0,
    )
    with pytest.raises(ManimVisionSchemaError):
        dispatcher.dispatch(bad, np.zeros(3), "shift(UP * 1.0)")
    assert buf.getvalue() == ""


def test_output_is_parseable_json() -> None:
    """Emitted telemetry must be strict JSON decodable."""
    buf = io.StringIO()
    dispatcher = TelemetryDispatcher(output_stream=buf, scene_name="JsonScene")
    dispatcher.dispatch(_sample_collision(), np.array([0.0, 0.0, 0.0]), "# noop")
    loaded = json.loads(buf.getvalue())
    assert isinstance(loaded, dict)


def test_colliding_entities_min_length() -> None:
    """``colliding_entities`` must always contain at least two string identifiers."""
    buf = io.StringIO()
    dispatcher = TelemetryDispatcher(output_stream=buf, scene_name="PairScene")
    payload = dispatcher.dispatch(_sample_collision(), np.ones(3), "shift(RIGHT * 1.0)")
    assert len(payload["colliding_entities"]) >= 2


def test_fix_suggestion_is_string() -> None:
    """``fix_suggestion`` must remain a plain string for downstream agents."""
    buf = io.StringIO()
    dispatcher = TelemetryDispatcher(output_stream=buf, scene_name="FixScene")
    payload = dispatcher.dispatch(_sample_collision(), np.zeros(3), "shift(LEFT * 0.5000)")
    assert isinstance(payload["fix_suggestion"], str)


def test_session_dedupe_skips_second_identical_pair(tmp_path) -> None:
    """Same semantic pair must not write two JSONL lines in one session."""
    jsonl = tmp_path / "d.jsonl"
    txt = tmp_path / "d.txt"
    dispatcher = TelemetryDispatcher(jsonl_path=jsonl, text_path=txt, scene_name="S")
    try:
        first = dispatcher.dispatch(
            _sample_collision(),
            np.array([0.0, 0.0, 0.0]),
            "# a",
            entity_labels=("A", "B"),
        )
        second = dispatcher.dispatch(
            _sample_collision(),
            np.array([0.0, 0.0, 0.0]),
            "# b",
            entity_labels=("A", "B"),
        )
        assert first is not None
        assert second is None
        assert len(jsonl.read_text().strip().splitlines()) == 1
    finally:
        dispatcher.close()


def test_file_output_jsonl_and_human_txt(tmp_path) -> None:
    """Default file mode must append one JSONL line and a text block, then close cleanly."""
    jsonl = tmp_path / "e.jsonl"
    txt = tmp_path / "e.txt"
    dispatcher = TelemetryDispatcher(jsonl_path=jsonl, text_path=txt, scene_name="Embeddings")
    try:
        dispatcher.dispatch(_sample_collision(), np.array([0.0, 0.0, 0.0]), "# noop")
    finally:
        dispatcher.close()
    line = jsonl.read_text(encoding="utf-8").strip()
    assert json.loads(line)["scene_name"] == "Embeddings"
    text_body = txt.read_text(encoding="utf-8")
    assert "OVERLAP" in text_body
    assert "entities:" in text_body


def test_write_check_digest_jsonl_and_summary_line_in_txt(tmp_path) -> None:
    """Legacy ``write_check_digest`` still appends a JSON line and a one-line text summary."""
    from manim_vision.telemetry.paths import check_digest_path_next_to_spatial_jsonl

    jsonl = tmp_path / "S_spatial.jsonl"
    txt = tmp_path / "S_log.txt"
    dpath = check_digest_path_next_to_spatial_jsonl(jsonl)
    assert dpath == tmp_path / "S_check_digest.jsonl"
    dispatcher = TelemetryDispatcher(
        jsonl_path=jsonl,
        text_path=txt,
        check_digest_path=dpath,
        scene_name="S",
    )
    try:
        dispatcher.write_check_digest(
            {
                "kind": "manim_vision_check_v1",
                "scene_name": "S",
                "raw_pair_hits": 5,
                "suppressed": {"below_min_area": 1},
                "actionable_merged": [{"pair": ["A", "B"]}],
            }
        )
    finally:
        dispatcher.close()
    lines = dpath.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["kind"] == "manim_vision_check_v1"
    assert len(payload["actionable_merged"]) == 1
    text_body = txt.read_text(encoding="utf-8")
    assert "manim-vision digest" in text_body
    assert "actionable=1" in text_body


def test_llm_summary_mode_flushes_collision_timeline() -> None:
    """LLM mode should flush finalized collision intervals with timing and geometry."""
    buf = io.StringIO()
    dispatcher = TelemetryDispatcher(output_stream=buf, scene_name="BinarySearchExplained", output_mode="llm")
    dispatcher.record_collision_event(
        {
            "objects": ['Text("Binary Search")', 'Text("Finding items fast in sorted data")'],
            "start_time": 0.0,
            "end_time": 2.5,
            "duration": 2.5,
            "peak_overlap_area": 0.1182,
            "peak_centroid": {"x": 0.02, "y": 3.41},
            "resolution_mtv": {"x": 0.0, "y": 0.1122, "z": 0.0},
            "fix_suggestion": "shift(UP * 0.1122)",
        }
    )
    dispatcher.close()
    payload = json.loads(buf.getvalue())
    assert payload["scene"] == "BinarySearchExplained"
    assert len(payload["collision_events"]) == 1
    event = payload["collision_events"][0]
    assert event["start_time"] == 0.0
    assert event["end_time"] == 2.5
    assert event["peak_centroid"] == {"x": 0.02, "y": 3.41}
    assert event["peak_overlap_area"] == pytest.approx(0.1182)
