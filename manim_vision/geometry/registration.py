"""Register the full mobject *family* so submobjects (not only group roots) are tracked."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from manim_vision.geometry.engine import PrecisionGeometryEngine


def register_mobject_families_in_engine(root: Any, engine: Any) -> None:
    """Register every :class:`VMobject` in ``root.get_family()``.

    The *root* instance passed to :meth:`~manim.scene.scene.Scene.add` is wrapped in
    :class:`ManimVisionMobjectProxy` so that ``deepcopy``-safe engine state stays on the
    proxy; all other family members are registered on the engine directly, because
    :class:`Scene` and parents still hold the original mobject references (submobject
    transforms never touch the group proxy on the way down).
    """
    from manim.mobject.types.vectorized_mobject import VMobject
    from manim_vision.proxy.mobject_proxy import ManimVisionMobjectProxy

    for member in root.get_family():
        if not isinstance(member, VMobject):
            continue
        if member is root:
            ManimVisionMobjectProxy(member, engine)
        else:
            engine.register(member)


def deregister_mobject_families_from_engine(root: Any, engine: Any) -> None:
    """Deregister all :class:`VMobject` in ``root.get_family()`` from the engine."""
    from manim.mobject.types.vectorized_mobject import VMobject

    for m in root.get_family():
        if isinstance(m, VMobject):
            engine.deregister(m)
