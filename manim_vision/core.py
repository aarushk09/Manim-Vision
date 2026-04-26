"""Public :class:`ManimVision` entry point for scene instrumentation."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

from manim_vision.exceptions import ManimVisionError
from manim_vision.overlay import maybe_install_overlay, render_overlay_subprocess
from manim_vision.proxy.scene_proxy import ManimVisionSceneMixin, _create_manim_vision_runtime_attrs
from manim_vision.telemetry.paths import default_check_digest_path

logger = logging.getLogger(__name__)


class ManimVision:
    """Activate Manim Vision spatial monitoring on a running Manim :class:`~manim.scene.scene.Scene`."""

    @classmethod
    def monitor(cls, scene: Any, output_mode: str = "llm") -> Any:
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
            output_mode: ``"llm"`` for JSON collision timelines, ``"human"`` for readable
                interval reports, or ``"silent"`` to suppress file/stdout output while
                keeping results available programmatically.

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

        effective_mode = "silent" if os.environ.get("MANIM_VISION_DISABLE_REPORT_WRITE", "").lower() in {
            "1",
            "true",
            "yes",
        } else output_mode

        for attr_name, attr_value in _create_manim_vision_runtime_attrs(user_scene_name, effective_mode).items():
            object.__setattr__(scene, attr_name, attr_value)

        merged = type(
            f"ManimVisionInstrumented{user_scene_name}",
            (ManimVisionSceneMixin, orig_cls),
            {},
        )
        scene.__class__ = merged
        maybe_install_overlay(scene)
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

    @classmethod
    def results(cls, scene: Any) -> dict[str, Any] | None:
        """Return the most recent collision timeline accumulated by the dispatcher."""
        dispatcher = scene.__dict__.get("_self_dispatcher")
        if dispatcher is None:
            return None
        return getattr(dispatcher, "results", None)

    @classmethod
    def render_overlay(
        cls,
        scene_or_script: Any,
        scene_name: str | None = None,
        *,
        script_path: str | Path | None = None,
        report_path: str | Path | None = None,
        quality_flag: str = "-ql",
        output_file: str | None = None,
    ) -> Path:
        """Re-render a scene with collision overlays highlighted in a separate video file."""
        from manim.scene.scene import Scene
        from manim import config as manim_config

        if isinstance(scene_or_script, Scene):
            resolved_scene_name = type(scene_or_script).__name__
            resolved_script = (
                Path(script_path)
                if script_path is not None
                else Path(sys.modules[type(scene_or_script).__module__].__file__ or "")
            )
        else:
            resolved_script = Path(scene_or_script)
            resolved_scene_name = scene_name or ""

        if not resolved_scene_name:
            raise ManimVisionError("render_overlay requires a scene name when called outside a Scene instance.")
        if not resolved_script.exists():
            raise ManimVisionError(f"render_overlay could not find script path: {resolved_script}")

        resolved_report_path = Path(report_path) if report_path is not None else _default_report_path_for_script(
            resolved_script,
            resolved_scene_name,
        )
        if not resolved_report_path.exists():
            raise ManimVisionError(
                f"Collision report not found for {resolved_scene_name}: {resolved_report_path}"
            )

        overlay_name = output_file or f"{resolved_scene_name}_collision_overlay"
        render_overlay_subprocess(
            script_path=resolved_script,
            scene_name=resolved_scene_name,
            report_path=resolved_report_path,
            output_file=overlay_name,
            quality_flag=quality_flag,
        )
        media_root = Path(str(getattr(manim_config, "media_dir", "media")))
        media_dir = media_root if media_root.is_absolute() else (resolved_script.parent / media_root).resolve()
        video_dir = media_dir / "videos" / resolved_script.stem / "480p15"
        return (video_dir / f"{overlay_name}.mp4").resolve()


def _default_report_path_for_script(script_path: Path, scene_name: str) -> Path:
    """Prefer the script-local Manim media directory before the process cwd default."""
    script_local = script_path.parent / "media" / "manim_vision" / f"{scene_name}_check_digest.jsonl"
    if script_local.exists():
        return script_local
    return default_check_digest_path(scene_name)
