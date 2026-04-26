"""Manim Vision — spatial intelligence for Manim (lazy public exports for Manim-free imports)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from manim_vision.exceptions import (
    ManimVisionError,
    ManimVisionGeometryError,
    ManimVisionProxyError,
    ManimVisionSchemaError,
)

__all__ = [
    "ManimVision",
    "ManimVisionError",
    "ManimVisionGeometryError",
    "ManimVisionProxyError",
    "ManimVisionSchemaError",
]


def __getattr__(name: str) -> Any:
    """Load Manim-dependent symbols only when requested."""
    if name == "ManimVision":
        from manim_vision.core import ManimVision

        return ManimVision
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:
    from manim_vision.core import ManimVision as ManimVision
