"""Minimum translation vectors, multi-body relaxation, and Manim fix strings."""

from __future__ import annotations

import logging
import math

import numpy as np
import shapely
from shapely import errors as shapely_errors

from manim_vision.geometry.engine import CollisionResult

logger = logging.getLogger(__name__)


class ConstraintSolver:
    """Computes MTV-style separation hints and optional force-based displacement estimates."""

    def calculate_mtv(self, result: CollisionResult) -> np.ndarray:
        """Compute a minimum translation style separation vector in world units.

        Convex shapes use a separating-axis (SAT) approximation on exterior rings.
        Concave shapes (convex hull area ratio above 1.02) use centroid displacement
        with depth ``sqrt(overlap_area)`` as documented conservative fallback (not full GJK/EPA).

        Args:
            result: A narrow-phase collision record with overlapping geometries.

        Returns:
            A length-3 NumPy vector ``[dx, dy, dz]`` with ``dz`` always ``0.0`` for 2D analysis.
        """
        try:
            area_a = float(result.geom_a.area)
            hull_a = float(result.geom_a.convex_hull.area)
        except (shapely_errors.TopologicalError, ValueError, shapely_errors.ShapelyError) as exc:
            logger.warning("MTV convexity probe failed; using centroid fallback: %s", exc)
            return self._mtv_centroid_fallback(result)

        if area_a <= 0.0 or not math.isfinite(area_a):
            return self._mtv_centroid_fallback(result)

        ratio = hull_a / area_a
        if ratio <= 1.02:
            mtv2 = self._mtv_sat(result.geom_a, result.geom_b)
            if mtv2 is not None:
                return np.array([mtv2[0], mtv2[1], 0.0], dtype=np.float64)
            return self._mtv_centroid_fallback(result)
        return self._mtv_centroid_fallback(result)

    def _mtv_centroid_fallback(self, result: CollisionResult) -> np.ndarray:
        """Return a centroid-based MTV for concave or degenerate SAT cases."""
        try:
            ca = np.array(result.geom_a.centroid.coords[0], dtype=np.float64)
            cb = np.array(result.geom_b.centroid.coords[0], dtype=np.float64)
        except (shapely_errors.TopologicalError, ValueError, shapely_errors.ShapelyError) as exc:
            logger.warning("Centroid MTV failed: %s", exc)
            return np.zeros(3, dtype=np.float64)
        direction = ca - cb
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            direction = np.array([1e-6, 0.0], dtype=np.float64)
            norm = float(np.linalg.norm(direction))
        direction = direction / norm
        depth = math.sqrt(max(float(result.overlap_area), 0.0))
        out = direction * depth
        return np.array([out[0], out[1], 0.0], dtype=np.float64)

    def _mtv_sat(self, geom_a: shapely.Geometry, geom_b: shapely.Geometry) -> np.ndarray | None:
        """Run SAT on exterior rings; return a 2D MTV or ``None`` if unsupported."""
        try:
            coords_a = self._exterior_coords(geom_a)
            coords_b = self._exterior_coords(geom_b)
        except (shapely_errors.TopologicalError, ValueError, shapely_errors.ShapelyError) as exc:
            logger.warning("SAT coordinate extraction failed: %s", exc)
            return None
        if len(coords_a) < 3 or len(coords_b) < 3:
            return None

        try:
            c_a = np.array(geom_a.centroid.coords[0][:2], dtype=np.float64)
            c_b = np.array(geom_b.centroid.coords[0][:2], dtype=np.float64)
        except (shapely_errors.TopologicalError, ValueError, shapely_errors.ShapelyError):
            c_a = np.mean(coords_a, axis=0)
            c_b = np.mean(coords_b, axis=0)

        axes: list[np.ndarray] = []
        for ring in (coords_a, coords_b):
            for i in range(len(ring) - 1):
                edge = ring[i + 1] - ring[i]
                n = np.array([-edge[1], edge[0]], dtype=np.float64)
                ln = float(np.linalg.norm(n))
                if ln < 1e-12:
                    continue
                axes.append(n / ln)

        best_depth = float("inf")
        best_axis = np.array([1.0, 0.0], dtype=np.float64)

        for axis in axes:
            min_a, max_a = self._project_interval(coords_a, axis)
            min_b, max_b = self._project_interval(coords_b, axis)
            overlap = min(max_a, max_b) - max(min_a, min_b)
            if overlap < 0:
                return None
            if overlap < best_depth:
                best_depth = overlap
                best_axis = axis

        push = best_axis * best_depth
        if np.dot(push, c_a - c_b) < 0:
            push = -push
        return push

    @staticmethod
    def _exterior_coords(geom: shapely.Geometry) -> np.ndarray:
        """Collect exterior ring coordinates for polygonal geometry."""
        if geom.geom_type == "Polygon":
            return np.asarray(geom.exterior.coords, dtype=np.float64)
        if geom.geom_type == "MultiPolygon":
            parts: list[np.ndarray] = []
            for p in geom.geoms:
                parts.append(np.asarray(p.exterior.coords, dtype=np.float64))
            if not parts:
                return np.empty((0, 2))
            return np.vstack(parts)
        hull = geom.convex_hull
        if hull.geom_type != "Polygon":
            return np.empty((0, 2))
        return np.asarray(hull.exterior.coords, dtype=np.float64)

    @staticmethod
    def _project_interval(points: np.ndarray, axis: np.ndarray) -> tuple[float, float]:
        """Project ``points`` onto ``axis`` and return ``(min, max)``."""
        dots = points @ axis
        return float(np.min(dots)), float(np.max(dots))

    def apply_force_relaxation(
        self,
        results: list[CollisionResult],
        iterations: int = 50,
        k_r: float = 8.5,
    ) -> dict[int, np.ndarray]:
        """Iteratively accumulate repulsive pseudo-forces between overlapping pairs.

        Args:
            results: Active collisions to relax.
            iterations: Maximum relaxation steps.
            k_r: Repulsion gain constant.

        Returns:
            Mapping ``mobject_id -> cumulative displacement`` as length-3 vectors.
        """
        involved: set[int] = set()
        for cr in results:
            involved.add(cr.mobject_a_id)
            involved.add(cr.mobject_b_id)

        displacements = {mid: np.zeros(3, dtype=np.float64) for mid in involved}

        for _ in range(iterations):
            before = {mid: displacements[mid].copy() for mid in involved}
            for cr in results:
                try:
                    mass_a = float(cr.geom_a.area)
                    mass_b = float(cr.geom_b.area)
                    centroid_a = np.array(cr.geom_a.centroid.coords[0], dtype=np.float64)
                    centroid_b = np.array(cr.geom_b.centroid.coords[0], dtype=np.float64)
                except (shapely_errors.TopologicalError, ValueError, shapely_errors.ShapelyError) as exc:
                    logger.warning("Force relaxation skipped a pair: %s", exc)
                    continue

                delta = centroid_a - centroid_b
                d = float(np.linalg.norm(delta[:2]))
                if d < 1e-9:
                    direction = np.array([1e-6, 0.0, 0.0], dtype=np.float64)
                    d = float(np.linalg.norm(direction))
                else:
                    direction = np.zeros(3, dtype=np.float64)
                    direction[:2] = delta[:2] / d

                f_r = k_r * (mass_a * mass_b) / max(d * d, 1e-18)
                force = f_r * direction
                displacements[cr.mobject_b_id] = displacements[cr.mobject_b_id] + force
                displacements[cr.mobject_a_id] = displacements[cr.mobject_a_id] - force

            step_delta = max(
                float(np.linalg.norm(displacements[mid] - before[mid])) for mid in involved
            )
            if step_delta < 1e-4:
                break

        return displacements

    def generate_fix_syntax(self, mtv: np.ndarray) -> str:
        """Translate an MTV vector into chained Manim ``shift`` calls.

        Args:
            mtv: Translation vector with up to three components.

        Returns:
            A dotted chain of ``shift(DIRECTION * mag)`` calls or a no-op comment.
        """
        dx = float(mtv[0])
        dy = float(mtv[1])
        parts: list[str] = []
        epsilon = 1e-4

        if abs(dy) > epsilon:
            direction = "UP" if dy > 0 else "DOWN"
            parts.append(f"shift({direction} * {abs(dy):.4f})")
        if abs(dx) > epsilon:
            direction = "RIGHT" if dx > 0 else "LEFT"
            parts.append(f"shift({direction} * {abs(dx):.4f})")

        if not parts:
            return "# No significant displacement required"
        return ".".join(parts)
