"""Broad-phase STRtree indexing and narrow-phase Shapely collision queries."""

from __future__ import annotations

import logging
import os
import threading
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING

import shapely
from shapely import STRtree
from shapely import errors as shapely_errors

from manim_vision.exceptions import ManimVisionGeometryError
from manim_vision.geometry.adapter import GeometryAdapter

if TYPE_CHECKING:
    from manim.mobject.types.vectorized_mobject import VMobject

logger = logging.getLogger(__name__)


def _verbose_geometry_skips() -> bool:
    """If true, log every skipped mobject at WARNING; default is DEBUG only."""
    return os.environ.get("MANIM_VISION_VERBOSE_GEO", "").lower() in (
        "1",
        "true",
        "yes",
    )


@dataclass(frozen=True)
class CollisionResult:
    """Immutable record of a 2D overlap between two monitored mobjects."""

    mobject_a_id: int
    mobject_b_id: int
    mobject_a_name: str
    mobject_b_name: str
    geom_a: shapely.Geometry
    geom_b: shapely.Geometry
    overlap_area: float
    overlap_geometry: shapely.Geometry


def _mobject_label(mobject: VMobject) -> str:
    """Build a stable string label ``ClassName_id`` for telemetry."""
    return f"{type(mobject).__name__}_{id(mobject)}"


class PrecisionGeometryEngine:
    """Maintains Shapely geometries for VMobjects and answers overlap queries."""

    def __init__(self, registry_lock: threading.Lock | None = None) -> None:
        """Create an engine with a dedicated adapter and logger.

        Args:
            registry_lock: Optional lock held around all registry reads and writes
                so a background collision pass cannot interleave with main-thread
                ``register`` / ``update`` / ``deregister`` calls.
        """
        self._registry: dict[int, tuple[VMobject, shapely.Geometry]] = {}
        self._adapter = GeometryAdapter()
        self._logger = logging.getLogger(__name__)
        self._registry_lock: AbstractContextManager = (
            registry_lock if registry_lock is not None else nullcontext()
        )

    def _registry_guard(self) -> AbstractContextManager:
        """Return the synchronization context for registry mutations and scans."""
        return self._registry_lock

    def register(self, mobject: VMobject) -> None:
        """Convert ``mobject`` to Shapely and store it under ``id(mobject)``.

        Args:
            mobject: The VMobject to track.

        Returns:
            None. Conversion failures are logged and skipped.
        """
        with self._registry_guard():
            try:
                geom = self._adapter.vmobject_to_polygon(mobject)
            except ManimVisionGeometryError as exc:
                if _verbose_geometry_skips():
                    self._logger.warning("Skipping registration for %s: %s", mobject, exc)
                else:
                    self._logger.debug("Skipping registration for %s: %s", mobject, exc)
                return
            self._registry[id(mobject)] = (mobject, geom)

    def update(self, mobject: VMobject) -> None:
        """Refresh the stored geometry after a spatial transform.

        Args:
            mobject: The VMobject whose outline changed.

        Returns:
            None. Missing registrations are ignored; conversion errors are logged.
        """
        with self._registry_guard():
            key = id(mobject)
            if key not in self._registry:
                return
            try:
                geom = self._adapter.vmobject_to_polygon(mobject)
            except ManimVisionGeometryError as exc:
                if _verbose_geometry_skips():
                    self._logger.warning("Skipping geometry update for %s: %s", mobject, exc)
                else:
                    self._logger.debug("Skipping geometry update for %s: %s", mobject, exc)
                return
            self._registry[key] = (mobject, geom)

    def deregister(self, mobject: VMobject) -> None:
        """Remove ``mobject`` from the registry if present.

        Args:
            mobject: The VMobject to stop tracking.

        Returns:
            None.
        """
        with self._registry_guard():
            self._registry.pop(id(mobject), None)

    def check_collisions(self) -> list[CollisionResult]:
        """Run broad-phase STRtree queries and narrow-phase DE-9IM-backed overlap tests.

        When a scene-scoped :class:`threading.Lock` was supplied at construction, callers
        must hold that lock for the duration of this method (e.g. the async collision
        pipeline) so registry snapshots stay consistent with telemetry emission.

        Returns:
            A list of :class:`CollisionResult` entries with positive overlap area.
        """
        if len(self._registry) < 2:
            return []

        keys = list(self._registry.keys())
        geometries = [self._registry[k][1] for k in keys]

        try:
            tree = STRtree(geometries)
        except (shapely_errors.TopologicalError, ValueError, shapely_errors.ShapelyError) as exc:
            self._logger.warning("STRtree construction failed: %s", exc)
            return []

        seen: set[tuple[int, int]] = set()
        results: list[CollisionResult] = []

        for local_i, geom_i in enumerate(geometries):
            key_i = keys[local_i]
            try:
                try:
                    raw_hits = tree.query(geom_i, predicate="intersects")
                except (shapely_errors.TopologicalError, ValueError, shapely_errors.ShapelyError):
                    self._logger.warning("STRtree query failed for index %s", local_i)
                    continue
                hits = raw_hits.tolist() if hasattr(raw_hits, "tolist") else list(raw_hits)
            except (shapely_errors.TopologicalError, ValueError, shapely_errors.ShapelyError) as exc:
                self._logger.warning("STRtree query failed: %s", exc)
                continue

            for local_j in hits:
                if local_j <= local_i:
                    continue
                key_j = keys[local_j]
                ka, kb = (key_i, key_j) if key_i < key_j else (key_j, key_i)
                norm = (ka, kb)
                if norm in seen:
                    continue
                seen.add(norm)

                mob_a, geom_a = self._registry[ka]
                mob_b, geom_b = self._registry[kb]

                try:
                    if geom_a.touches(geom_b):
                        continue
                    if not geom_a.intersects(geom_b):
                        continue
                    de9im = geom_a.relate(geom_b)
                    if len(de9im) < 1 or de9im[0] not in {"0", "1", "2"}:
                        continue
                except (shapely_errors.TopologicalError, ValueError, shapely_errors.ShapelyError) as exc:
                    self._logger.warning(
                        "Topological predicate failed for pair (%s, %s): %s",
                        ka,
                        kb,
                        exc,
                    )
                    continue

                try:
                    overlap_geometry = geom_a.intersection(geom_b)
                    overlap_area = float(overlap_geometry.area)
                except (shapely_errors.TopologicalError, ValueError, shapely_errors.ShapelyError) as exc:
                    self._logger.warning("intersection failed for pair (%s, %s): %s", ka, kb, exc)
                    continue

                if overlap_area <= 0.0:
                    continue

                results.append(
                    CollisionResult(
                        mobject_a_id=ka,
                        mobject_b_id=kb,
                        mobject_a_name=_mobject_label(mob_a),
                        mobject_b_name=_mobject_label(mob_b),
                        geom_a=geom_a,
                        geom_b=geom_b,
                        overlap_area=overlap_area,
                        overlap_geometry=overlap_geometry,
                    )
                )

        return results
