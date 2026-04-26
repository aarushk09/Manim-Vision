"""Public :class:`ManimVision` entry point for scene instrumentation."""

from __future__ import annotations

import logging
from typing import Any

from manim_vision.exceptions import ManimVisionError
from manim_vision.proxy.scene_proxy import ManimVisionSceneMixin, _create_manim_vision_runtime_attrs

logger = logging.getLogger(__name__)


class ManimVision:
    """Activate Manim Vision spatial monitoring on a running Manim :class:`~manim.scene.scene.Scene`."""

    @classmethod
    def monitor(cls, scene: Any) -> Any:
        """Instrument ``scene`` so ``add`` / ``play`` / ``remove`` run collision analysis.

        ``ManimVision.monitor(self)`` does not replace the ``self`` reference in the caller's
        stack frame (that is impossible in Python). Instead it:

        1. Attaches ``_self_engine``, ``_self_solver``, ``_self_dispatcher``,
           ``_self_lock``, and ``_self_executor`` (single-worker pool) on the scene
           instance so collision analysis can run off the animation thread.
        2. Rebinds ``scene.__class__`` to a freshly built type
           ``type("ManimVisionInstrumented{UserScene}", (ManimVisionSceneMixin, UserScene), {})``.
           Because ``ManimVisionSceneMixin`` is listed first, ``Scene.add``, ``Scene.play``, and
           ``Scene.remove`` resolve to the mixin overrides while all other methods and
           attributes continue to resolve through the user's original subclass.

        Args:
            scene: The live ``Scene`` instance (typically ``self`` inside ``construct``).

        Returns:
            The same ``scene`` instance, now with Manim Vision hooks installed.

        Raises:
            ManimVisionError: If ``scene`` is not a Manim :class:`~manim.scene.scene.Scene`.
        """
        from manim.scene.scene import Scene

        if not isinstance(scene, Scene):
            raise ManimVisionError(f"ManimVision.monitor() requires a Manim Scene instance. Got: {type(scene)}")

        orig_cls = type(scene)
        user_scene_name = orig_cls.__name__

        for attr_name, attr_value in _create_manim_vision_runtime_attrs(user_scene_name).items():
            object.__setattr__(scene, attr_name, attr_value)

        merged = type(
            f"ManimVisionInstrumented{user_scene_name}",
            (ManimVisionSceneMixin, orig_cls),
            {},
        )
        scene.__class__ = merged
        logger.info("Manim Vision monitoring enabled for scene class %s", user_scene_name)
        return scene

    @classmethod
    def shutdown(cls, scene: Any) -> None:
        """Flush pending asynchronous collision checks before the scene tears down.

        Call ``ManimVision.shutdown(self)`` at the end of ``construct()`` (or after the last
        ``add`` / ``play``) so the collision worker finishes and the thread pool is
        cleanly stopped before Manim exits.

        Args:
            scene: The instrumented :class:`~manim.scene.scene.Scene` instance.

        Returns:
            None.
        """
        shutdown_fn = getattr(scene, "shutdown", None)
        if callable(shutdown_fn):
            shutdown_fn()
            return
        executor = scene.__dict__.get("_self_executor")
        if executor is not None:
            executor.shutdown(wait=True)
        dispatcher = scene.__dict__.get("_self_dispatcher")
        if dispatcher is not None and hasattr(dispatcher, "close"):
            dispatcher.close()
