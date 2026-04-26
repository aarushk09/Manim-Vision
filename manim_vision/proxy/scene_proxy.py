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
    SceneSemanticResolver,
    min_reportable_overlap_area,
    per_pair_jsonl_enabled,
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
    active_events: set[tuple[int, int]] = getattr(scene, "_self_active_collision_events")

    with lock:
        # Animations only move submobject points; scene.mobjects’ families must be
        # re-baked to Shapely before testing pairs.
        engine.resync_scene_mobjects(list(scene.mobjects))
        raw_hits = engine.check_collisions()
        resolver = SceneSemanticResolver(scene)
        if not raw_hits:
            active_events.clear()
            return

        min_a = min_reportable_overlap_area()
        work: list[CollisionResult] = []
        n_tiny = 0
        n_glyphs = 0
        n_intent = 0
        for cr in raw_hits:
            if cr.overlap_area < min_a:
                n_tiny += 1
                continue
            m_a = engine._registry[cr.mobject_a_id][0]
            m_b = engine._registry[cr.mobject_b_id][0]
            if resolver.is_pair_internal_glyphs_same_text(m_a, m_b):
                n_glyphs += 1
                continue
            if resolver.is_intentional_layout_pair(m_a, m_b):
                n_intent += 1
                continue
            work.append(cr)

        suppressed = {
            "below_min_area": n_tiny,
            "same_text_glyphs": n_glyphs,
            "intentional_layout": n_intent,
        }

        if not work:
            active_events.clear()
            return

        solver.apply_force_relaxation(work)

        clusters: dict[tuple[int, int], list[CollisionResult]] = {}
        for cr in work:
            m_a = engine._registry[cr.mobject_a_id][0]
            m_b = engine._registry[cr.mobject_b_id][0]
            event_key = resolver.event_key(m_a, m_b)
            clusters.setdefault(event_key, []).append(cr)

        current_events = set(clusters)
        new_events = current_events - active_events
        active_events.clear()
        active_events.update(current_events)
        if not new_events:
            return

        per_pair = per_pair_jsonl_enabled()
        n_new = 0
        n_duped = 0
        actionable: list[dict[str, Any]] = []

        for event_key in sorted(new_events):
            group = clusters[event_key]
            best = max(group, key=lambda c: c.overlap_area)
            m_a = engine._registry[best.mobject_a_id][0]
            m_b = engine._registry[best.mobject_b_id][0]
            mtv = solver.calculate_mtv(best)
            fix_syntax = solver.generate_fix_syntax(mtv)
            la, lb = resolver.pair_labels(m_a, m_b)
            if per_pair:
                out = dispatcher.dispatch(
                    best,
                    mtv,
                    fix_syntax,
                    scene_name=scene_name,
                    entity_labels=(la, lb),
                )
                if out is None:
                    n_duped += 1
                else:
                    n_new += 1
            actionable.append(
                {
                    "pair": [la, lb],
                    "merged_from": len(group),
                    "max_overlap_area": float(best.overlap_area),
                    "resolution_mtv": {
                        "x": float(mtv[0]),
                        "y": float(mtv[1]),
                        "z": float(mtv[2]) if len(mtv) > 2 else 0.0,
                    },
                    "fix_suggestion": fix_syntax,
                }
            )

        dispatcher.write_check_digest(
            {
                "kind": "manim_vision_check_v1",
                "scene_name": scene_name,
                "new_event_count": len(actionable),
                "raw_pair_hits": len(raw_hits),
                "suppressed": suppressed,
                "actionable_merged": actionable,
            }
        )

        n_action = len(actionable)
        if per_pair and n_new:
            logger.info(
                "Manim Vision: %d new per-pair event(s) for %s (digest merged %d pair label(s) from %d raw; "
                "raw=%d, suppressed %r).",
                n_new,
                scene_name,
                n_action,
                len(work),
                len(raw_hits),
                suppressed,
            )
        elif n_action and not per_pair:
            logger.info(
                "Manim Vision: digest only — %d new actionable event(s) for %s "
                "(set MANIM_VISION_PER_PAIR_JSONL=1 for legacy one-line-per-pair; raw=%d, suppressed %r).",
                n_action,
                scene_name,
                len(raw_hits),
                suppressed,
            )
        elif per_pair and n_action:
            logger.debug(
                "Manim Vision: 0 new per-pair lines (all %d merged pair(s) duplicate vs session) for %s",
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


def _create_manim_vision_runtime_attrs(scene_name: str, output_mode: str = "llm") -> dict[str, Any]:
    """Build shared runtime objects for scene instrumentation.

    Args:
        scene_name: Manim scene class name used in telemetry payloads.

    Returns:
        Mapping of attribute names to values for ``__dict__`` / ``object.__setattr__``.
    """
    lock = threading.RLock()
    engine = PrecisionGeometryEngine(registry_lock=lock)
    solver = ConstraintSolver()
    dispatcher = TelemetryDispatcher(scene_name=scene_name, output_mode=output_mode)
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="manim-vision-collision")
    return {
        "_self_engine": engine,
        "_self_solver": solver,
        "_self_dispatcher": dispatcher,
        "_self_lock": lock,
        "_self_executor": executor,
        "_self_active_collision_events": set(),
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
