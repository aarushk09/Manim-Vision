"""Scene-level interception for add/play/remove and collision telemetry."""

from __future__ import annotations

import copy
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
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


@dataclass
class ActiveCollisionEvent:
    """Live interval for one semantic pair until the overlap resolves."""

    objects: tuple[str, str]
    start_time: float
    latest_time: float
    peak_overlap_area: float
    peak_centroid: tuple[float, float]
    resolution_mtv: tuple[float, float, float]
    fix_suggestion: str


def _scene_time_seconds(scene: Any) -> float:
    """Return Manim's current animation clock in seconds."""
    renderer = getattr(scene, "renderer", None)
    if renderer is not None and hasattr(renderer, "time"):
        try:
            return float(renderer.time)
        except (TypeError, ValueError):
            pass
    fallback = getattr(scene, "time", 0.0)
    try:
        return float(fallback)
    except (TypeError, ValueError):
        return 0.0


def _event_payload(event: ActiveCollisionEvent, *, end_time: float) -> dict[str, Any]:
    """Build one finalized collision event record for the dispatcher."""
    return {
        "objects": list(event.objects),
        "start_time": float(event.start_time),
        "end_time": float(end_time),
        "duration": max(0.0, float(end_time) - float(event.start_time)),
        "peak_overlap_area": float(event.peak_overlap_area),
        "peak_centroid": {
            "x": float(event.peak_centroid[0]),
            "y": float(event.peak_centroid[1]),
        },
        "resolution_mtv": {
            "x": float(event.resolution_mtv[0]),
            "y": float(event.resolution_mtv[1]),
            "z": float(event.resolution_mtv[2]),
        },
        "fix_suggestion": event.fix_suggestion,
    }


def _close_finished_events(
    active_events: dict[tuple[int, int], ActiveCollisionEvent],
    dispatcher: TelemetryDispatcher,
    open_now: set[tuple[int, int]],
    timestamp: float,
) -> None:
    """Finalize every active pair that is no longer overlapping."""
    for event_key in sorted(set(active_events) - open_now):
        dispatcher.record_collision_event(_event_payload(active_events.pop(event_key), end_time=timestamp))


def _flush_open_collision_events(scene: Any) -> None:
    """Finalize intervals still open when monitoring shuts down."""
    dispatcher = getattr(scene, "_self_dispatcher", None)
    active_events: dict[tuple[int, int], ActiveCollisionEvent] = getattr(
        scene,
        "_self_active_collision_events",
        {},
    )
    if dispatcher is None or not active_events:
        return
    timestamp = _scene_time_seconds(scene)
    for event_key in sorted(active_events):
        dispatcher.record_collision_event(_event_payload(active_events[event_key], end_time=timestamp))
    active_events.clear()


def _execute_collision_check(scene: Any) -> None:
    """Run collision detection, interval tracking, and telemetry under the scene lock."""
    lock = getattr(scene, "_self_lock")
    engine = getattr(scene, "_self_engine")
    solver = getattr(scene, "_self_solver")
    dispatcher = getattr(scene, "_self_dispatcher")
    scene_name = getattr(scene, "_manim_vision_scene_class_name")
    active_events: dict[tuple[int, int], ActiveCollisionEvent] = getattr(
        scene,
        "_self_active_collision_events",
    )

    with lock:
        timestamp = _scene_time_seconds(scene)
        # Animations move submobject points under stable scene roots, so the engine
        # needs a fresh geometry pass before any event lifecycle decision is made.
        engine.resync_scene_mobjects(list(scene.mobjects))
        raw_hits = engine.check_collisions()
        resolver = SceneSemanticResolver(scene)
        if not raw_hits:
            _close_finished_events(active_events, dispatcher, set(), timestamp)
            return

        min_area = min_reportable_overlap_area()
        work: list[CollisionResult] = []
        n_tiny = 0
        n_glyphs = 0
        n_intent = 0
        for collision in raw_hits:
            if collision.overlap_area < min_area:
                n_tiny += 1
                continue
            mobject_a = engine._registry[collision.mobject_a_id][0]
            mobject_b = engine._registry[collision.mobject_b_id][0]
            if resolver.owner(mobject_a) is resolver.owner(mobject_b):
                n_glyphs += 1
                continue
            if resolver.is_pair_internal_glyphs_same_text(mobject_a, mobject_b):
                n_glyphs += 1
                continue
            if resolver.is_intentional_layout_pair(mobject_a, mobject_b):
                n_intent += 1
                continue
            work.append(collision)

        suppressed = {
            "below_min_area": n_tiny,
            "same_text_glyphs": n_glyphs,
            "intentional_layout": n_intent,
        }

        if not work:
            _close_finished_events(active_events, dispatcher, set(), timestamp)
            return

        solver.apply_force_relaxation(work)

        clusters: dict[tuple[int, int], list[CollisionResult]] = {}
        for collision in work:
            mobject_a = engine._registry[collision.mobject_a_id][0]
            mobject_b = engine._registry[collision.mobject_b_id][0]
            event_key = resolver.event_key(mobject_a, mobject_b)
            clusters.setdefault(event_key, []).append(collision)

        current_events = set(clusters)
        _close_finished_events(active_events, dispatcher, current_events, timestamp)

        per_pair = per_pair_jsonl_enabled()
        n_new = 0
        n_duped = 0
        n_opened = 0

        for event_key in sorted(current_events):
            group = clusters[event_key]
            best = max(group, key=lambda collision: collision.overlap_area)
            mobject_a = engine._registry[best.mobject_a_id][0]
            mobject_b = engine._registry[best.mobject_b_id][0]
            mtv = solver.calculate_mtv(best)
            fix_syntax = solver.generate_fix_syntax(mtv)
            label_a, label_b = resolver.pair_labels(mobject_a, mobject_b)
            mtv_tuple = (
                float(mtv[0]),
                float(mtv[1]),
                float(mtv[2]) if len(mtv) > 2 else 0.0,
            )

            if event_key not in active_events:
                active_events[event_key] = ActiveCollisionEvent(
                    objects=(label_a, label_b),
                    start_time=timestamp,
                    latest_time=timestamp,
                    peak_overlap_area=float(best.overlap_area),
                    peak_centroid=best.overlap_centroid,
                    resolution_mtv=mtv_tuple,
                    fix_suggestion=fix_syntax,
                )
                n_opened += 1
            else:
                event = active_events[event_key]
                event.latest_time = timestamp
                event.objects = (label_a, label_b)
                if float(best.overlap_area) >= float(event.peak_overlap_area):
                    event.peak_overlap_area = float(best.overlap_area)
                    event.peak_centroid = best.overlap_centroid
                    event.resolution_mtv = mtv_tuple
                    event.fix_suggestion = fix_syntax

            if per_pair:
                dispatched = dispatcher.dispatch(
                    best,
                    mtv,
                    fix_syntax,
                    scene_name=scene_name,
                    entity_labels=(label_a, label_b),
                )
                if dispatched is None:
                    n_duped += 1
                else:
                    n_new += 1

        if per_pair and n_new:
            logger.info(
                "Manim Vision: %d new per-pair event(s) for %s (%d interval(s) open; raw=%d, suppressed %r).",
                n_new,
                scene_name,
                len(current_events),
                len(raw_hits),
                suppressed,
            )
        elif current_events and not per_pair:
            logger.info(
                "Manim Vision: tracking %d open collision interval(s) for %s (opened %d this pass; raw=%d, suppressed %r).",
                len(current_events),
                scene_name,
                n_opened,
                len(raw_hits),
                suppressed,
            )
        elif per_pair and current_events:
            logger.debug(
                "Manim Vision: 0 new per-pair lines (all %d interval(s) duplicate vs session) for %s",
                n_duped,
                scene_name,
            )


def _submit_collision_check(scene: Any) -> None:
    """Schedule :meth:`_run_collision_check` on the scene's single-worker executor."""
    executor: ThreadPoolExecutor = getattr(scene, "_self_executor")
    executor.submit(scene._run_collision_check)


def _create_manim_vision_runtime_attrs(scene_name: str, output_mode: str = "llm") -> dict[str, Any]:
    """Build shared runtime objects for scene instrumentation."""
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
        "_self_active_collision_events": {},
        "_manim_vision_scene_class_name": scene_name,
    }


class ManimVisionSceneMixin:
    """Mixin merged ahead of the user's ``Scene`` subclass by :meth:`manim_vision.core.ManimVision.monitor`."""

    def add(self, *mobjects: Any) -> Any:
        """Register geometry for every :class:`VMobject` in each added mobject's family."""
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
        result = super().remove(*mobjects)
        _submit_collision_check(self)
        return result

    def play(self, *animations: Any, **kwargs: Any) -> Any:
        """Delegate to Manim's ``Scene.play`` and schedule collision analysis."""
        result = super().play(*animations, **kwargs)
        _submit_collision_check(self)
        return result

    def _run_collision_check(self) -> None:
        """Synchronous collision pipeline entry point (invoked on the executor worker)."""
        _execute_collision_check(self)

    def shutdown(self) -> None:
        """Wait for queued collision work to finish and shut down the worker pool."""
        executor: ThreadPoolExecutor | None = getattr(self, "_self_executor", None)
        if executor is not None:
            executor.shutdown(wait=True)
        _flush_open_collision_events(self)
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
        result = self.__wrapped__.remove(*mobjects)
        _submit_collision_check(self)
        return result

    def play(self, *animations: Any, **kwargs: Any) -> Any:
        """Play animations on the wrapped scene then schedule collision analysis."""
        result = self.__wrapped__.play(*animations, **kwargs)
        _submit_collision_check(self)
        return result

    def _run_collision_check(self) -> None:
        """Synchronous collision pipeline entry point (invoked on the executor worker)."""
        _execute_collision_check(self)

    def shutdown(self) -> None:
        """Wait for queued collision work to finish and shut down the worker pool."""
        executor: ThreadPoolExecutor | None = getattr(self, "_self_executor", None)
        if executor is not None:
            executor.shutdown(wait=True)
        _flush_open_collision_events(self)
        dispatcher = getattr(self, "_self_dispatcher", None)
        if dispatcher is not None and hasattr(dispatcher, "close"):
            dispatcher.close()
