"""Scene-level interception for add/play/remove and collision telemetry."""

from __future__ import annotations

import copy
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import wrapt

from manim_vision.geometry.engine import CollisionResult, PrecisionGeometryEngine
from manim_vision.geometry.registration import (
    deregister_mobject_families_from_engine,
    register_mobject_families_in_engine,
)
from manim_vision.semantic import (
    is_pair_internal_glyphs_same_text,
    min_reportable_overlap_area,
    semantic_label,
)
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
    lock = getattr(scene, "_self_lock")
    engine = getattr(scene, "_self_engine")
    solver = getattr(scene, "_self_solver")
    dispatcher = getattr(scene, "_self_dispatcher")
    scene_name = getattr(scene, "_manim_vision_scene_class_name")

    with lock:
        # Animations only move submobject points; scene.mobjects’ families must be
        # re-baked to Shapely before testing pairs.
        engine.resync_scene_mobjects(list(scene.mobjects))
        raw_hits = engine.check_collisions()
        if not raw_hits:
            return

        min_a = min_reportable_overlap_area()
        filtered: list[CollisionResult] = []
        n_tiny = 0
        n_glyphs = 0
        for cr in raw_hits:
            if cr.overlap_area < min_a:
                n_tiny += 1
                continue
            m_a = engine._registry[cr.mobject_a_id][0]
            m_b = engine._registry[cr.mobject_b_id][0]
            if is_pair_internal_glyphs_same_text(m_a, m_b, scene):
                n_glyphs += 1
                continue
            filtered.append(cr)

        if not filtered:
            return

        solver.apply_force_relaxation(filtered)

        n_new = 0
        n_duped = 0
        for cr in filtered:
            m_a = engine._registry[cr.mobject_a_id][0]
            m_b = engine._registry[cr.mobject_b_id][0]
            mtv = solver.calculate_mtv(cr)
            fix_syntax = solver.generate_fix_syntax(mtv)
            labels = (semantic_label(m_a, scene), semantic_label(m_b, scene))
            out = dispatcher.dispatch(
                cr,
                mtv,
                fix_syntax,
                scene_name=scene_name,
                entity_labels=labels,
            )
            if out is None:
                n_duped += 1
            else:
                n_new += 1

        if n_new:
            logger.info(
                "Manim Vision: %d new unique overlap report(s) for scene %s (check had %d raw, "
                "%d below min area, %d same-text internal, %d duplicate vs earlier in session).",
                n_new,
                scene_name,
                len(raw_hits),
                n_tiny,
                n_glyphs,
                n_duped,
            )
        elif filtered:
            logger.debug(
                "Manim Vision: 0 new reports (all %d pair(s) already in session) for %s",
                n_duped,
                scene_name,
            )


def _submit_collision_check(scene: Any) -> None:
    """Schedule :meth:`_run_collision_check` on the scene's single-worker executor.

    Args:
        scene: Instrumented scene or proxy with ``_self_executor`` populated.
    """
    executor: ThreadPoolExecutor = getattr(scene, "_self_executor")
    executor.submit(scene._run_collision_check)


def _create_manim_vision_runtime_attrs(scene_name: str) -> dict[str, Any]:
    """Build shared runtime objects for scene instrumentation.

    Args:
        scene_name: Manim scene class name used in telemetry payloads.

    Returns:
        Mapping of attribute names to values for ``__dict__`` / ``object.__setattr__``.
    """
    lock = threading.RLock()
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
        """Register geometry for every :class:`VMobject` in each added mobject’s family."""
        engine = self._self_engine
        for mob in mobjects:
            register_mobject_families_in_engine(mob, engine)
        result = super().add(*mobjects)
        _submit_collision_check(self)
        return result

    def remove(self, *mobjects: Any) -> Any:
        """Deregister all family VMobjects then delegate to ``Scene.remove``."""
        engine = self._self_engine
        for mob in mobjects:
            deregister_mobject_families_from_engine(mob, engine)
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
        executor: ThreadPoolExecutor | None = getattr(self, "_self_executor", None)
        if executor is not None:
            executor.shutdown(wait=True)
        dispatcher = getattr(self, "_self_dispatcher", None)
        if dispatcher is not None and hasattr(dispatcher, "close"):
            dispatcher.close()


class ManimVisionSceneProxy(wrapt.ObjectProxy):
    """Top-level wrapt proxy that mirrors :class:`ManimVisionSceneMixin` for wrapped scenes."""

    def __init__(self, wrapped_scene: Any) -> None:
        """Attach engines, solver, dispatcher, lock, and executor to ``wrapped_scene``."""
        super().__init__(wrapped_scene)
        scene_name = type(wrapped_scene).__name__
        for key, value in _create_manim_vision_runtime_attrs(scene_name).items():
            # Use wrapt's helper: plain ``object.__setattr__`` fails on ObjectProxy slottage;
            # ``__setattr__`` would route non-``_self_`` names onto the wrapped scene.
            self.__self_setattr__(key, value)

    def __deepcopy__(self, clone_from_id: Any) -> Any:
        """Deep-copy the wrapped scene only, omitting proxy state (lock, engine, etc.)."""
        return copy.deepcopy(self.__wrapped__, clone_from_id)

    def add(self, *mobjects: Any) -> Any:
        """Wrap additions with geometry registration and schedule a collision check."""
        engine = self._self_engine
        for mob in mobjects:
            register_mobject_families_in_engine(mob, engine)
        result = self.__wrapped__.add(*mobjects)
        _submit_collision_check(self)
        return result

    def remove(self, *mobjects: Any) -> Any:
        """Remove mobjects from the scene and registry."""
        engine = self._self_engine
        for mob in mobjects:
            deregister_mobject_families_from_engine(mob, engine)
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
        executor: ThreadPoolExecutor | None = getattr(self, "_self_executor", None)
        if executor is not None:
            executor.shutdown(wait=True)
        dispatcher = getattr(self, "_self_dispatcher", None)
        if dispatcher is not None and hasattr(dispatcher, "close"):
            dispatcher.close()
