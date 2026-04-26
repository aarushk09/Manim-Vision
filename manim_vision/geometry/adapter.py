"""Convert Manim VMobjects to Shapely geometries for planar collision analysis."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import shapely
from shapely import errors as shapely_errors
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPolygon,
    Polygon,
)

from manim_vision.exceptions import ManimVisionGeometryError

if TYPE_CHECKING:
    from manim.mobject.types.vectorized_mobject import VMobject

logger = logging.getLogger(__name__)


class GeometryAdapter:
    """Adapts Manim :class:`VMobject` point data into validated Shapely geometries."""

    def vmobject_to_polygon(
        self,
        mobject: VMobject,
        precision_grid: float = 1e-6,
    ) -> shapely.Geometry:
        """Convert a VMobject's stroke/fill outline into a Shapely geometry.

        Args:
            mobject: The Manim vectorized mobject to convert.
            precision_grid: Grid size passed to :func:`shapely.set_precision`.

        Returns:
            A ``Polygon``, ``MultiPolygon``, or ``LineString`` representing the outline.

        Raises:
            ManimVisionGeometryError: If point data is insufficient or geometry cannot be made valid.
        """
        from manim import config

        points = np.asarray(mobject.points, dtype=np.float64)
        if points.size == 0 or len(points) < 2:
            raise ManimVisionGeometryError(
                f"VMobject '{mobject}' has insufficient points for geometry conversion "
                f"(got {len(points)})."
            )

        frame_w = float(getattr(config, "frame_width", 14.0))
        pix_w = float(getattr(config, "pixel_width", 1920))
        pixel_size = frame_w / max(pix_w, 1.0)

        if len(points) == 2:
            ring = self._project_2d(points)
            if self._all_points_coincident(ring):
                msg = f"VMobject '{mobject}' degenerates to a single location."
                logger.warning(msg)
                raise ManimVisionGeometryError(msg)
            geom = LineString(ring)
            return self._finalize_geometry(geom, precision_grid, mobject)

        degree = self._detect_bezier_degree(points, mobject)
        subpaths = self._extract_subpaths(points, mobject)

        polygons: list[Polygon] = []
        open_rings: list[np.ndarray] = []

        for sub in subpaths:
            sub_arr = np.asarray(sub, dtype=np.float64)
            if len(sub_arr) < 2:
                continue
            if self._all_points_coincident(sub_arr):
                msg = f"VMobject '{mobject}' contains a degenerate subpath; skipping it."
                logger.warning(msg)
                continue
            sampled = self._discretize_subpath(sub_arr, degree, pixel_size)
            ring_2d = self._project_2d(sampled)
            if len(ring_2d) < 2:
                continue
            if self._is_subpath_closed(sub_arr, mobject):
                if len(ring_2d) < 3:
                    continue
                poly = Polygon(ring_2d)
                polygons.append(poly)
            else:
                open_rings.append(ring_2d)

        geom: shapely.Geometry
        if polygons and open_rings:
            parts: list[shapely.Geometry] = [*polygons]
            for ring in open_rings:
                if len(ring) >= 2:
                    parts.append(LineString(ring))
            geom = GeometryCollection(parts)
        elif polygons:
            geom = polygons[0] if len(polygons) == 1 else MultiPolygon(polygons)
        elif open_rings:
            lines = [LineString(r) for r in open_rings if len(r) >= 2]
            if not lines:
                raise ManimVisionGeometryError(
                    f"VMobject '{mobject}' produced no drawable geometry after discretization."
                )
            geom = lines[0] if len(lines) == 1 else MultiLineString(lines)
        else:
            raise ManimVisionGeometryError(
                f"VMobject '{mobject}' produced no drawable geometry after discretization."
            )

        if geom.is_empty:
            raise ManimVisionGeometryError(f"VMobject '{mobject}' produced an empty geometry.")

        return self._finalize_geometry(geom, precision_grid, mobject)

    @staticmethod
    def _all_points_coincident(pts: np.ndarray) -> bool:
        """Return True if every row in ``pts`` is identical within a tight tolerance."""
        if len(pts) == 0:
            return True
        return bool(np.all(np.linalg.norm(pts - pts[0], axis=1) < 1e-12))

    @staticmethod
    def _project_2d(pts: np.ndarray) -> np.ndarray:
        """Project Nx3 points to the XY plane (Manim camera projection)."""
        return np.asarray(pts[:, :2], dtype=np.float64)

    @staticmethod
    def _detect_bezier_degree(points: np.ndarray, mobject: Any) -> int:
        """Infer quadratic (2) or cubic (3) Bézier layout from renderer metadata or stride.

        If ``(len(points) - 1) % 3 == 0`` with at least four points, Manim's default cubic
        chain is assumed (stride of one anchor every three points after the first curve).
        Otherwise quadratic (stride two) is assumed when ``(len(points) - 1) % 2 == 0``.

        Args:
            points: Raw ``(N, 3)`` anchor/handle samples from the VMobject.
            mobject: The VMobject being inspected (for ``renderer_type`` / ``n_points_per_cubic_curve``).

        Returns:
            ``2`` for quadratic splines, ``3`` for cubic splines.
        """
        rt = getattr(mobject, "renderer_type", None)
        if rt is not None:
            label = str(rt).lower()
            if "opengl" in label or label.endswith("gl"):
                return 2
            return 3
        n = len(points)
        if n >= 4 and (n - 1) % 3 == 0:
            return 3
        if n >= 3 and (n - 1) % 2 == 0:
            return 2
        return 3

    @staticmethod
    def _extract_subpaths(points: np.ndarray, mobject: Any) -> list[np.ndarray]:
        """Split ``points`` into Bézier chains using Manim's subpath convention when available."""
        getter = getattr(mobject, "get_subpaths", None)
        if callable(getter):
            try:
                raw_subs = getter()
                return [np.asarray(sp, dtype=np.float64) for sp in raw_subs]
            except (TypeError, ValueError, IndexError) as exc:
                logger.warning("get_subpaths failed (%s); falling back to anchor scan.", exc)

        boundaries = [0]
        for i in range(len(points) - 1):
            if np.allclose(points[i], points[i + 1], atol=1e-9, rtol=0.0):
                boundaries.append(i + 1)
        boundaries.append(len(points))
        out: list[np.ndarray] = []
        for a, b in zip(boundaries[:-1], boundaries[1:], strict=False):
            if b > a:
                out.append(points[a:b])
        return out if out else [points]

    @staticmethod
    def _is_subpath_closed(subpath: np.ndarray, mobject: Any) -> bool:
        """Return whether ``subpath`` should be treated as a closed ring."""
        closer = getattr(mobject, "consider_points_equals", None)
        if len(subpath) < 2:
            return False
        if callable(closer):
            try:
                return bool(closer(subpath[0], subpath[-1]))
            except (TypeError, ValueError):
                pass
        return bool(np.allclose(subpath[0], subpath[-1], atol=1e-6, rtol=0.0))

    def _discretize_subpath(
        self,
        subpath: np.ndarray,
        degree: int,
        pixel_size: float,
    ) -> np.ndarray:
        """Sample all Bézier segments in ``subpath`` into a dense polyline."""
        if degree == 3:
            stride = 3
            width = 4
        else:
            stride = 2
            width = 3
        samples: list[np.ndarray] = []
        n = len(subpath)
        if n < width:
            return subpath
        idx = 0
        while idx + width <= n:
            segment = subpath[idx : idx + width]
            chord = float(np.linalg.norm(segment[-1] - segment[0]))
            n_samples = max(4, int(chord / max(pixel_size, 1e-12)) + 1)
            ts = np.linspace(0.0, 1.0, n_samples)
            for t in ts:
                samples.append(self._de_casteljau(segment, float(t)))
            idx += stride
        if not samples:
            return subpath
        return np.vstack(samples)

    @staticmethod
    def _de_casteljau(control_points: np.ndarray, t: float) -> np.ndarray:
        """Evaluate a Bézier curve at parameter ``t`` using De Casteljau's algorithm.

        Args:
            control_points: ``(degree+1, dim)`` control points (``dim`` is 2 or 3).
            t: Interpolation parameter in ``[0, 1]``.

        Returns:
            The evaluated point with the same dimensionality as ``control_points``.
        """
        pts = np.asarray(control_points, dtype=np.float64).copy()
        n = len(pts)
        for r in range(1, n):
            for i in range(n - r):
                pts[i] = (1.0 - t) * pts[i] + t * pts[i + 1]
        return pts[0]

    def _finalize_geometry(
        self,
        geometry: shapely.Geometry,
        precision_grid: float,
        mobject: Any,
    ) -> shapely.Geometry:
        """Apply precision grid, validate, and repair when possible."""
        try:
            geom = shapely.set_precision(geometry, grid_size=precision_grid)
        except (shapely_errors.TopologicalError, ValueError, shapely_errors.ShapelyError) as exc:
            logger.warning(
                "set_precision failed for VMobject '%s': %s",
                mobject,
                exc,
            )
            geom = geometry

        try:
            valid = bool(geom.is_valid)
        except (shapely_errors.TopologicalError, ValueError, shapely_errors.ShapelyError) as exc:
            logger.warning("is_valid check failed for VMobject '%s': %s", mobject, exc)
            valid = False

        if not valid:
            try:
                geom = shapely.make_valid(geom)
            except (shapely_errors.TopologicalError, ValueError, shapely_errors.ShapelyError) as exc:
                logger.warning("make_valid failed for VMobject '%s': %s", mobject, exc)
                raise ManimVisionGeometryError(
                    f"VMobject '{mobject}' could not be repaired into a valid geometry."
                ) from exc

        try:
            still_bad = not geom.is_valid or geom.is_empty
        except (shapely_errors.TopologicalError, ValueError, shapely_errors.ShapelyError) as exc:
            logger.warning("post-repair validation failed for VMobject '%s': %s", mobject, exc)
            still_bad = True

        if still_bad:
            raise ManimVisionGeometryError(
                f"VMobject '{mobject}' remains invalid or empty after make_valid."
            )
        return geom
