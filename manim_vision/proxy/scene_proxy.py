"""Scene-level interception for add/play/remove and collision telemetry."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import wrapt

from manim_vision.geometry.engine import PrecisionGeometryEngine
from manim_vision.proxy.mobject_proxy import ManimVisionMobjectProxy
from manim_vision.solver.constraint import ConstraintSolver
from manim_vision.telemetry.dispatcher import TelemetryDispatcher

logger = logging.getLogger(__name__)


def _execute_collision_check(scene: Any) -> None:
    """Run collision detection, relaxation, and telemetry under the scene registry lock.

    The caller must invoke this from the collision worker thread. The scene's
    ``_self_lock`` serializes this body against main-thread registry mutations.

    Args:
        scene: The instrumented scene or scene proxy carrying ManimVision runtime attributes.
    """
    lock = scene.__dict__["_self_lock"]
    engine = scene.__dict__["_self_engine"]
    solver = scene.__dict__["_self_solver"]
    dispatcher = scene.__dict__["_self_dispatcher"]
    scene_name = scene.__dict__["_manim_vision_scene_class_name"]

    with lock:
        collision_results = engine.check_collisions()
        if not collision_results:
            return

        solver.apply_force_relaxation(collision_results)

        for cr in collision_results:
            mtv = solver.calculate_mtv(cr)
            fix_syntax = solver.generate_fix_syntax(mtv)
            dispatcher.dispatch(cr, mtv, fix_syntax, scene_name=scene_name)


def _submit_collision_check(scene: Any) -> None:
    """Schedule :meth:`_run_collision_check` on the scene's single-worker executor.

    Args:
        scene: Instrumented scene or proxy with ``_self_executor`` populated.
    """
    executor: ThreadPoolExecutor = scene.__dict__["_self_executor"]
    executor.submit(scene._run_collision_check)


def _create_manim_vision_runtime_attrs(scene_name: str) -> dict[str, Any]:
    """Build shared runtime objects for scene instrumentation.

    Args:
        scene_name: Manim scene class name used in telemetry payloads.

    Returns:
        Mapping of attribute names to values for ``__dict__`` / ``object.__setattr__``.
    """
    lock = threading.Lock()
    engine = PrecisionGeometryEngine(registry_lock=lock)
    solver = ConstraintSolver()
    dispatcher = TelemetryDispatcher(scene_name=scene_name)
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="manim-vision-collision")
    return {
        "_self_engine": engine,
        "_self_solver": solver,
        "_self_dispatcher": dispatcher,
        "_self_lock": lock,
        "_self_executor": executor,
        "_manim_vision_scene_class_name": scene_name,
    }


class ManimVisionSceneMixin:
    """Mixin merged ahead of the user's ``Scene`` subclass by :meth:`manim_vision.core.ManimVision.monitor`."""

    def add(self, *mobjects: Any) -> Any:
        """Register geometry proxies then delegate to Manim's ``Scene.add``."""
        engine = self.__dict__["_self_engine"]
        for mob in mobjects:
            ManimVisionMobjectProxy(mob, engine)
        result = super().add(*mobjects)
        _submit_collision_check(self)
        return result

    def remove(self, *mobjects: Any) -> Any:
        """Deregister tracked mobjects then delegate to ``Scene.remove``."""
        engine = self.__dict__["_self_engine"]
        for mob in mobjects:
            engine.deregister(mob)
        return super().remove(*mobjects)

    def play(self, *animations: Any, **kwargs: Any) -> Any:
        """Delegate to Manim's ``Scene.play`` and schedule collision analysis."""
        result = super().play(*animations, **kwargs)
        _submit_collision_check(self)
        return result

    def _run_collision_check(self) -> None:
        """Synchronous collision pipeline entry point (invoked on the executor worker)."""
        _execute_collision_check(self)

    def shutdown(self) -> None:
        """Wait for queued collision work to finish and shut down the worker pool.

        Long-running scenes should call :meth:`manim_vision.core.ManimVision.shutdown` (or this
        method) at the end of ``construct()`` so pending spatial checks complete
        before the renderer finalizes.

        Returns:
            None.
        """
        executor: ThreadPoolExecutor | None = self.__dict__.get("_self_executor")
        if executor is not None:
            executor.shutdown(wait=True)


class ManimVisionSceneProxy(wrapt.ObjectProxy):
    """Top-level wrapt proxy that mirrors :class:`ManimVisionSceneMixin` for wrapped scenes."""

    def __init__(self, wrapped_scene: Any) -> None:
        """Attach engines, solver, dispatcher, lock, and executor to ``wrapped_scene``."""
        super().__init__(wrapped_scene)
        scene_name = type(wrapped_scene).__name__
        for key, value in _create_manim_vision_runtime_attrs(scene_name).items():
            self.__dict__[key] = value

    def add(self, *mobjects: Any) -> Any:
        """Wrap additions with geometry registration and schedule a collision check."""
        engine = self.__dict__["_self_engine"]
        for mob in mobjects:
            ManimVisionMobjectProxy(mob, engine)
        result = self.__wrapped__.add(*mobjects)
        _submit_collision_check(self)
        return result

    def remove(self, *mobjects: Any) -> Any:
        """Remove mobjects from the scene and registry."""
        engine = self.__dict__["_self_engine"]
        for mob in mobjects:
            engine.deregister(mob)
        return self.__wrapped__.remove(*mobjects)

    def play(self, *animations: Any, **kwargs: Any) -> Any:
        """Play animations on the wrapped scene then schedule collision analysis."""
        result = self.__wrapped__.play(*animations, **kwargs)
        _submit_collision_check(self)
        return result

    def _run_collision_check(self) -> None:
        """Synchronous collision pipeline entry point (invoked on the executor worker)."""
        _execute_collision_check(self)

    def shutdown(self) -> None:
        """Wait for queued collision work to finish and shut down the worker pool.

        Long-running scenes should call this method or :meth:`manim_vision.core.ManimVision.shutdown`
        at the end of ``construct()`` so pending spatial checks complete before the
        renderer finalizes.

        Returns:
            None.
        """
        executor: ThreadPoolExecutor | None = self.__dict__.get("_self_executor")
        if executor is not None:
            executor.shutdown(wait=True)
