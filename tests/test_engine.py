"""Tests for :mod:`manim_vision.geometry.engine`."""

from __future__ import annotations

import logging

import pytest
from manim import RIGHT, Circle, Mobject, Rectangle, Square, Text, VGroup

from manim_vision.geometry.engine import PrecisionGeometryEngine
from manim_vision.geometry.registration import iter_trackable_family_members, register_mobject_families_in_engine


def test_no_collision_on_separated_objects() -> None:
    """Separated circles must not produce any collision records."""
    eng = PrecisionGeometryEngine()
    a = Circle(radius=0.5)
    b = Circle(radius=0.5).shift([10.0, 0.0, 0.0])
    eng.register(a)
    eng.register(b)
    assert eng.check_collisions() == []


def test_collision_detected_on_overlap() -> None:
    """Fully overlapping circles must register exactly one overlapping pair."""
    eng = PrecisionGeometryEngine()
    a = Circle(radius=1.0)
    b = Circle(radius=1.0)
    eng.register(a)
    eng.register(b)
    hits = eng.check_collisions()
    assert len(hits) == 1
    assert hits[0].overlap_area > 0.0
    assert len(hits[0].overlap_centroid) == 2


def test_touching_boundaries_not_collision() -> None:
    """Edge-adjacent squares must touch without positive-area overlap."""
    eng = PrecisionGeometryEngine()
    s1 = Square(side_length=2.0)
    s2 = Square(side_length=2.0).shift([2.0, 0.0, 0.0])
    eng.register(s1)
    eng.register(s2)
    assert eng.check_collisions() == []


def test_deregister_removes_from_checks() -> None:
    """Removing one participant from the registry must eliminate its collisions."""
    eng = PrecisionGeometryEngine()
    a = Circle(radius=1.0)
    b = Circle(radius=1.0)
    eng.register(a)
    eng.register(b)
    assert len(eng.check_collisions()) == 1
    eng.deregister(b)
    assert eng.check_collisions() == []


def test_n_squared_collision_count() -> None:
    """Five mutually overlapping discs must yield ten unique unordered pairs."""
    eng = PrecisionGeometryEngine()
    circles = [Circle(radius=2.0) for _ in range(5)]
    for c in circles:
        eng.register(c)
    hits = eng.check_collisions()
    assert len(hits) == 10


def test_update_recalculates_geometry() -> None:
    """After ``update``, shifted geometry must clear a collision that existed before."""
    eng = PrecisionGeometryEngine()
    a = Circle(radius=1.0)
    b = Circle(radius=1.0)
    eng.register(a)
    eng.register(b)
    assert len(eng.check_collisions()) == 1
    b.shift([10.0, 0.0, 0.0])
    eng.update(b)
    assert eng.check_collisions() == []


def test_vgroup_family_registers_submobjects_and_detects_overlap() -> None:
    """Each VMobject in a :class:`VGroup` must be a separate Shapely entry for ST queries."""
    eng = PrecisionGeometryEngine()
    a = Square(side_length=0.4)
    b = Square(side_length=0.4)
    b.next_to(a, RIGHT, buff=-0.15)
    g = VGroup(a, b)
    register_mobject_families_in_engine(g, eng)
    assert id(a) in eng._registry and id(b) in eng._registry
    hits = eng.check_collisions()
    assert len(hits) >= 1


def test_text_registers_each_character_as_separate_component() -> None:
    """A ``Text`` object should contribute one tracked component per visible character."""
    text = Text("AB")
    components = list(iter_trackable_family_members(text))
    assert len(components) == 2
    assert all(type(component).__name__ == "VMobjectFromSVGPath" for component in components)


def test_known_group_overlap_scene_matches_human_collision_count() -> None:
    """A wide rectangle covering two grouped squares should yield exactly two overlaps."""
    eng = PrecisionGeometryEngine()
    left = Square(side_length=1.0).shift([-0.75, 0.0, 0.0])
    right = Square(side_length=1.0).shift([0.75, 0.0, 0.0])
    pair = VGroup(left, right)
    sweep = Rectangle(width=2.2, height=0.8)
    register_mobject_families_in_engine(pair, eng)
    eng.register(sweep)
    hits = eng.check_collisions()
    assert len(hits) == 2


def test_register_empty_mobject_logs_debug_not_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Mobjects with no stroke points (e.g. unlaid Text) should not flood WARNING-level logs."""
    caplog.set_level(logging.DEBUG)
    eng = PrecisionGeometryEngine()
    eng.register(Mobject())
    assert not any(
        r.levelno == logging.WARNING and "Skipping registration" in r.getMessage()
        for r in caplog.records
    )
    assert any(
        r.levelno == logging.DEBUG and "Skipping registration" in r.getMessage() for r in caplog.records
    )
