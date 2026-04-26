"""Wrapt proxy that forwards VMobject calls and refreshes geometry after transforms."""

from __future__ import annotations

import logging
from typing import Any, Callable

import wrapt

from manim_vision.geometry.engine import PrecisionGeometryEngine

logger = logging.getLogger(__name__)


class ManimVisionMobjectProxy(wrapt.ObjectProxy):
    """Non-destructive proxy that refreshes the geometry engine after spatial edits.

    Spatial methods listed in :attr:`_SPATIAL_METHODS` trigger a post-call
    :meth:`~manim_vision.geometry.engine.PrecisionGeometryEngine.update` on the wrapped
    VMobject without mutating Manim internals.
    """

    _SPATIAL_METHODS = frozenset(
        {
            "shift",
            "scale",
            "rotate",
            "move_to",
            "next_to",
            "align_to",
            "set_x",
            "set_y",
            "set_z",
            "stretch",
            "apply_matrix",
            "apply_function",
        }
    )

    def __init__(self, wrapped: Any, engine: PrecisionGeometryEngine) -> None:
        """Wrap ``wrapped`` and register it with ``engine``.

        Args:
            wrapped: Original Manim VMobject instance.
            engine: Owning :class:`~manim_vision.geometry.engine.PrecisionGeometryEngine`.
        """
        super().__init__(wrapped)
        self.__dict__["_self_engine"] = engine
        engine.register(wrapped)

    def __getattr__(self, name: str) -> Any:
        """Resolve attributes on the wrapped object, intercepting spatial mutators."""
        attr = getattr(self.__wrapped__, name)
        if name in self._SPATIAL_METHODS and callable(attr):
            return self._make_intercepted_call(attr)
        return attr

    def _make_intercepted_call(self, method: Callable[..., Any]) -> Callable[..., Any]:
        """Return a wrapper that triggers geometry updates after the real call."""

        def intercepted(*args: Any, **kwargs: Any) -> Any:
            result = method(*args, **kwargs)
            engine = self.__dict__["_self_engine"]
            engine.update(self.__wrapped__)
            return result

        return intercepted
