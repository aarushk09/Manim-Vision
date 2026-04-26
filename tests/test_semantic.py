"""Tests for :mod:`manim_vision.semantic` layout and pairing helpers."""

from __future__ import annotations

from types import SimpleNamespace

from manim import Text

from manim_vision.semantic import SceneSemanticResolver, is_intentional_layout_pair, is_strict_submobject


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


def test_text_labels_use_content_and_duplicate_suffixes() -> None:
    """Two scene texts with the same content must still receive unique, readable labels."""

    class Text:
        def __init__(self, text: str) -> None:
            self.text = text
            self.submobjects = ()

        def get_family(self):
            return (self,)

    a = Text("mid")
    b = Text("mid")
    scene = SimpleNamespace(mobjects=[a, b])
    resolver = SceneSemanticResolver(scene)
    assert resolver.label(a) == 'Text("mid")[1]'
    assert resolver.label(b) == 'Text("mid")[2]'


def test_mathtex_labels_use_tex_content() -> None:
    """Math text should be labeled with its TeX content rather than raw ids."""

    class MathTex:
        def __init__(self, tex_string: str) -> None:
            self.tex_string = tex_string
            self.submobjects = ()

        def get_family(self):
            return (self,)

    expr = MathTex(r"\log n")
    scene = SimpleNamespace(mobjects=[expr])
    resolver = SceneSemanticResolver(scene)
    assert resolver.label(expr) == "MathTex('\\\\log n')"


def test_text_components_get_character_labels() -> None:
    """Tracked text glyphs should be labeled by parent content and character index."""
    text = Text("AB")
    scene = SimpleNamespace(mobjects=[text])
    resolver = SceneSemanticResolver(scene)
    assert resolver.label(text[0]) == 'Text("AB").char[0]'
    assert resolver.label(text[1]) == 'Text("AB").char[1]'


def test_path_artifact_near_same_text_is_treated_as_intentional() -> None:
    """Generic path fragments near the same text they came from should not become real collisions."""

    class Artifact:
        submobjects = ()

        def get_family(self):
            return (self,)

        def get_center(self):
            return (0.0, 0.0, 0.0)

    class TextLike:
        def __init__(self, text: str) -> None:
            self.original_text = text
            self.submobjects = ()

        def get_family(self):
            return (self,)

        def get_center(self):
            return (0.0, 0.0, 0.0)

    artifact = Artifact()
    artifact.__class__.__name__ = "VMobjectFromSVGPath"
    text = TextLike("Caption")
    text.__class__.__name__ = "Text"
    scene = SimpleNamespace(mobjects=[text, artifact])
    resolver = SceneSemanticResolver(scene)
    assert resolver.is_intentional_layout_pair(artifact, text) is True
