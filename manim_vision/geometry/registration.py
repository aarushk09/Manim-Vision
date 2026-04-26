"""Register the full mobject *family* so submobjects (not only group roots) are tracked."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from manim_vision.geometry.engine import PrecisionGeometryEngine

_TEXT_LIKE_NAMES = frozenset({"Text", "MarkupText", "MathTex", "Tex", "SingleStringMathTex"})
_SKIPPED_INTERNAL_TYPES = frozenset({"VMobjectFromSVGPath", "VectorizedPoint"})


def _text_ancestor_for(member: Any, root: Any) -> Any | None:
    for ancestor in root.get_family():
        if type(ancestor).__name__ not in _TEXT_LIKE_NAMES:
            continue
        family_getter = getattr(ancestor, "get_family", None)
        family = family_getter() if callable(family_getter) else (ancestor,)
        if member is ancestor or member in family:
            return ancestor
    return None


def iter_trackable_family_members(root: Any) -> Iterator[Any]:
    """Yield only the family members that are useful collision participants.

    Generic internal SVG path fragments and vectorized points create massive noise
    for composite objects. We keep text glyph geometry because text roots have no
    points of their own, but skip other internal fragments unless the user added
    them directly as the root object.
    """
    from manim.mobject.types.vectorized_mobject import VMobject

    seen: set[int] = set()
    for member in root.get_family():
        if not isinstance(member, VMobject):
            continue
        name = type(member).__name__
        if name == "VectorizedPoint":
            continue
        if member is not root and name == "VMobjectFromSVGPath" and _text_ancestor_for(member, root) is None:
            continue
        key = id(member)
        if key in seen:
            continue
        seen.add(key)
        yield member


def register_mobject_families_in_engine(root: Any, engine: Any) -> None:
    """Register every :class:`VMobject` in ``root.get_family()``.

    The *root* instance passed to :meth:`~manim.scene.scene.Scene.add` is wrapped in
    :class:`ManimVisionMobjectProxy` so that ``deepcopy``-safe engine state stays on the
    proxy; all other family members are registered on the engine directly, because
    :class:`Scene` and parents still hold the original mobject references (submobject
    transforms never touch the group proxy on the way down).
    """
    from manim_vision.proxy.mobject_proxy import ManimVisionMobjectProxy

    for member in iter_trackable_family_members(root):
        if member is root:
            ManimVisionMobjectProxy(member, engine)
        else:
            engine.register(member)


def deregister_mobject_families_from_engine(root: Any, engine: Any) -> None:
    """Deregister all :class:`VMobject` in ``root.get_family()`` from the engine."""
    for member in iter_trackable_family_members(root):
        engine.deregister(member)
