"""Tests for :mod:`manim_vision.core`."""

from __future__ import annotations

import pytest
from manim import Circle, FadeIn
from manim.scene.scene import Scene

from manim_vision.core import ManimVision
from manim_vision.exceptions import ManimVisionError
from manim_vision.proxy.scene_proxy import ManimVisionSceneMixin


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
