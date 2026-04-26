"""Register the lowest meaningful drawable VMobject components for collision checks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from manim_vision.geometry.engine import PrecisionGeometryEngine

_SKIPPED_INTERNAL_TYPES = frozenset({"VectorizedPoint"})


def _is_drawable_component(member: Any) -> bool:
    """Return whether ``member`` has enough point data to be visually meaningful."""
    points = getattr(member, "points", None)
    try:
        return points is not None and len(points) >= 2
    except TypeError:
        return False


def iter_trackable_family_members(root: Any) -> Iterator[Any]:
    """Yield the deepest drawable VMobjects that correspond to visible scene parts.

    The collision system should operate on visual components such as individual text
    characters, MathTex glyph paths, or members of a ``VGroup``. We therefore recurse
    until we hit drawable leaves, and only fall back to the parent object when a branch
    has no drawable descendants of its own.
    """
    from manim.mobject.types.vectorized_mobject import VMobject

    seen: set[int] = set()

    def visit(member: Any) -> Iterator[Any]:
        if not isinstance(member, VMobject):
            return
        if getattr(member, "_manim_vision_ignore", False):
            return
        if type(member).__name__ in _SKIPPED_INTERNAL_TYPES:
            return

        children = [
            child
            for child in list(getattr(member, "submobjects", ()) or ())
            if isinstance(child, VMobject) and type(child).__name__ not in _SKIPPED_INTERNAL_TYPES
        ]

        yielded_child = False
        for child in children:
            child_yielded = False
            for descendant in visit(child):
                child_yielded = True
                yielded_child = True
                yield descendant
            if not child_yielded and _is_drawable_component(child):
                key = id(child)
                if key not in seen:
                    seen.add(key)
                    yielded_child = True
                    yield child

        if yielded_child:
            return

        if _is_drawable_component(member):
            key = id(member)
            if key not in seen:
                seen.add(key)
                yield member

    yield from visit(root)


def register_mobject_families_in_engine(root: Any, engine: Any) -> None:
    """Register every trackable component of ``root`` in the geometry engine."""
    from manim_vision.proxy.mobject_proxy import ManimVisionMobjectProxy

    proxy_attached = False
    for member in iter_trackable_family_members(root):
        if member is root and not proxy_attached:
            ManimVisionMobjectProxy(member, engine)
            proxy_attached = True
        else:
            engine.register(member)
    if root is not None and not proxy_attached:
        # Even if ``root`` itself is not a collision participant, its transforms still need
        # to trigger engine resyncs for any registered descendants.
        ManimVisionMobjectProxy(root, engine)


def deregister_mobject_families_from_engine(root: Any, engine: Any) -> None:
    """Deregister all tracked drawable components under ``root`` from the engine."""
    for member in iter_trackable_family_members(root):
        engine.deregister(member)
