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
        a = dispatcher.dispatch(_sample_collision(), np.array([0.0, 0.0, 0.0]), "# a", entity_labels=("A", "B"))
        b = dispatcher.dispatch(_sample_collision(), np.array([0.0, 0.0, 0.0]), "# b", entity_labels=("A", "B"))
        assert a is not None
        assert b is None
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
