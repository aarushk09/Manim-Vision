"""Tests for wrapt-based scene and mobject proxies."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

from manim import Circle, VMobject
from manim.scene.scene import Scene

from manim_vision.geometry.engine import PrecisionGeometryEngine
from manim_vision.proxy.mobject_proxy import ManimVisionMobjectProxy
from manim_vision.proxy.scene_proxy import ManimVisionSceneProxy


def test_isinstance_preserved() -> None:
    """``ManimVisionMobjectProxy`` must preserve ``isinstance`` checks against ``VMobject``."""
    engine = PrecisionGeometryEngine()
    circle = Circle()
    proxy = ManimVisionMobjectProxy(circle, engine)
    assert isinstance(proxy, VMobject)


def test_shift_triggers_engine_update() -> None:
    """Spatial ``shift`` must invoke :meth:`~manim_vision.geometry.engine.PrecisionGeometryEngine.update` once."""
    engine = MagicMock()
    circle = Circle()
    proxy = ManimVisionMobjectProxy(circle, engine)
    proxy.shift([0.1, 0.0, 0.0])
    engine.update.assert_called_once_with(circle)


def test_non_spatial_method_passthrough() -> None:
    """Non-spatial mutators must not trigger geometry engine updates."""
    engine = MagicMock()
    circle = Circle()
    proxy = ManimVisionMobjectProxy(circle, engine)
    proxy.set_color("#FFFFFF")
    engine.update.assert_not_called()


def test_scene_add_wraps_mobject() -> None:
    """``ManimVisionSceneProxy.add`` must register raw mobjects with the internal engine."""
    scene = Scene()
    proxy = ManimVisionSceneProxy(scene)
    try:
        c = Circle()
        proxy.add(c)
        engine = proxy.__dict__["_self_engine"]
        assert id(c) in engine._registry
    finally:
        proxy.shutdown()


def test_scene_remove_deregisters() -> None:
    """``remove`` must clear registry entries created by ``add``."""
    scene = Scene()
    proxy = ManimVisionSceneProxy(scene)
    try:
        c = Circle()
        proxy.add(c)
        engine = proxy.__dict__["_self_engine"]
        proxy.remove(c)
        assert id(c) not in engine._registry
    finally:
        proxy.shutdown()


def test_add_does_not_block() -> None:
    """``add`` must return before the collision worker finishes long-running checks."""

    def _slow_collision_check(self: object) -> None:
        time.sleep(0.5)

    scene = Scene()
    with patch.object(ManimVisionSceneProxy, "_run_collision_check", _slow_collision_check):
        proxy = ManimVisionSceneProxy(scene)
        try:
            start = time.monotonic()
            proxy.add(Circle())
            elapsed = time.monotonic() - start
            assert elapsed < 0.05
        finally:
            proxy.shutdown()


def test_collision_check_runs_eventually() -> None:
    """After ``add`` returns, ``shutdown`` must wait for the submitted collision job."""
    scene = Scene()
    with patch.object(ManimVisionSceneProxy, "_run_collision_check", MagicMock()) as mocked:
        proxy = ManimVisionSceneProxy(scene)
        proxy.add(Circle())
        proxy.shutdown()
        mocked.assert_called_once()


def _noop_collision_check(self: object) -> None:
    """Collision no-op for concurrency tests that focus on registry locking."""
    return None


def test_registry_lock_prevents_race() -> None:
    """Concurrent ``add`` calls must not corrupt the engine registry."""
    scene = Scene()
    with patch.object(ManimVisionSceneProxy, "_run_collision_check", _noop_collision_check):
        proxy = ManimVisionSceneProxy(scene)
        errors: list[BaseException] = []

        def burst() -> None:
            try:
                for _ in range(100):
                    proxy.add(Circle(radius=0.05))
            except RuntimeError as exc:
                errors.append(exc)

        try:
            t1 = threading.Thread(target=burst)
            t2 = threading.Thread(target=burst)
            t1.start()
            t2.start()
            t1.join()
            t2.join()
            assert not errors
        finally:
            proxy.shutdown()
