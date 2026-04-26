"""Tests for :mod:`manim_vision.core`."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from manim import Circle, FadeIn
from manim.scene.scene import Scene

from manim_vision.core import ManimVision
from manim_vision.exceptions import ManimVisionError
from manim_vision.proxy.scene_proxy import _execute_collision_check
from manim_vision.proxy.scene_proxy import ManimVisionSceneMixin
from manim_vision.telemetry.dispatcher import TelemetryDispatcher


def test_monitor_installs_mixin_mro() -> None:
    """``ManimVision.monitor`` must prepend :class:`ManimVisionSceneMixin` so ``add`` is instrumented."""
    scene = Scene()
    try:
        ManimVision.monitor(scene)
        assert ManimVisionSceneMixin in type(scene).__mro__
        assert type(scene).__name__.startswith("ManimVisionInstrumented")
    finally:
        ManimVision.shutdown(scene)


def test_monitor_rejects_non_scene() -> None:
    """``ManimVision.monitor`` must raise :class:`ManimVisionError` for objects that are not scenes."""
    with pytest.raises(ManimVisionError, match="Manim Scene"):
        ManimVision.monitor(object())


def test_monitored_mobject_fadein_begin_does_not_raise_on_deepcopy() -> None:
    """Creation-style animations must not hit ``_thread.lock`` when copying a proxied mobject (regression)."""
    scene = Scene()
    ManimVision.monitor(scene)
    try:
        mob = Circle()
        scene.add(mob)
        anim = FadeIn(mob)
        anim._setup_scene(scene)
        anim.begin()
    finally:
        ManimVision.shutdown(scene)


def test_persistent_collision_emits_once_until_objects_separate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A continuous overlap should become one event; separating and re-colliding creates a second."""
    monkeypatch.setenv("MANIM_VISION_REPORT_DIR", str(tmp_path))
    scene = Scene()
    a = Circle(radius=1.0)
    b = Circle(radius=1.0)
    scene.mobjects = [a, b]
    scene._self_lock = threading.RLock()
    scene._self_engine = None
    scene._self_solver = None
    scene._self_dispatcher = None
    scene._self_active_collision_events = set()
    scene._manim_vision_scene_class_name = "TestScene"

    from manim_vision.geometry.engine import PrecisionGeometryEngine
    from manim_vision.solver.constraint import ConstraintSolver

    engine = PrecisionGeometryEngine(registry_lock=scene._self_lock)
    engine.register(a)
    engine.register(b)
    scene._self_engine = engine
    scene._self_solver = ConstraintSolver()
    scene._self_dispatcher = TelemetryDispatcher(scene_name="TestScene", output_mode="llm")

    _execute_collision_check(scene)
    _execute_collision_check(scene)
    b.shift([5.0, 0.0, 0.0])
    engine.update(b)
    _execute_collision_check(scene)
    b.shift([-5.0, 0.0, 0.0])
    engine.update(b)
    _execute_collision_check(scene)
    scene._self_dispatcher.close()

    results = scene._self_dispatcher.results
    assert results is not None
    assert results["events"] == 2


def test_silent_mode_keeps_results_in_memory_without_writing_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Silent mode should suppress file output while leaving a programmatic summary available."""
    monkeypatch.setenv("MANIM_VISION_REPORT_DIR", str(tmp_path))
    scene = Scene()
    ManimVision.monitor(scene, output_mode="silent")
    try:
        scene.add(Circle(radius=1.0), Circle(radius=1.0))
        ManimVision.shutdown(scene)
        results = ManimVision.results(scene)
        assert results is not None
        assert results["events"] >= 1
        assert list(tmp_path.iterdir()) == []
    finally:
        ManimVision.shutdown(scene)
