"""Non-destructive wrapt proxies for Scene and VMobject interception."""

from __future__ import annotations

from manim_vision.proxy.mobject_proxy import ManimVisionMobjectProxy
from manim_vision.proxy.scene_proxy import ManimVisionSceneProxy

__all__ = ["ManimVisionMobjectProxy", "ManimVisionSceneProxy"]
