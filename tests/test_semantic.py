"""Tests for :mod:`manim_vision.semantic` layout and pairing helpers."""

from __future__ import annotations

from types import SimpleNamespace

from manim_vision.semantic import is_intentional_layout_pair, is_strict_submobject


def test_strict_submobject() -> None:
    """Descendant mobjects (label inside group) are detected."""

    class Inner:
        def get_family(self):
            return (self,)

    inner = Inner()

    class Outer:
        def get_family(self):
            return (self, inner)

    out = Outer()
    assert is_strict_submobject(out, inner) is True
    assert is_strict_submobject(inner, out) is False


def test_sibling_shape_text_tiles() -> None:
    """A small ``VGroup`` with a shape + ``Text`` as direct children counts as a tile, not a bug."""
    TText = type("Text", (), {})
    TSq = type("Square", (), {})
    t, s = TText(), TSq()

    class VGroup:
        def __init__(self) -> None:
            self.submobjects = (s, t)

    g = VGroup()
    g.get_family = lambda: (g, t, s)  # type: ignore[assignment, misc]

    root = SimpleNamespace()
    root.get_family = lambda: (root, g)  # type: ignore[assignment, misc]

    scene = SimpleNamespace(mobjects=[root])
    assert is_intentional_layout_pair(s, t, scene) is True
    assert is_intentional_layout_pair(t, s, scene) is True
