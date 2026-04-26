"""Tests for :mod:`manim_vision.geometry.engine`."""

from __future__ import annotations

import logging

import pytest
from manim import Circle, Mobject, Square

from manim_vision.geometry.engine import PrecisionGeometryEngine


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
