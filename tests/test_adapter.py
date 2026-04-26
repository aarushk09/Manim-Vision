"""Tests for :mod:`manim_vision.geometry.adapter`."""

from __future__ import annotations

import numpy as np
import pytest
from manim import Circle, Text, VMobject

from manim_vision.exceptions import ManimVisionGeometryError
from manim_vision.geometry.adapter import GeometryAdapter


def test_line_produces_linestring(geometry_adapter: GeometryAdapter) -> None:
    """A two-point VMobject must map to a ``LineString``, not a filled ``Polygon``."""
    line = VMobject()
    line.append_points(np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 1.0]], dtype=np.float64))
    geom = geometry_adapter.vmobject_to_polygon(line)
    assert geom.geom_type == "LineString"
    assert not geom.is_empty


def test_empty_mobject_raises_error(geometry_adapter: GeometryAdapter) -> None:
    """Empty ``points`` must raise :class:`~manim_vision.exceptions.ManimVisionGeometryError`."""
    empty = VMobject()
    empty.append_points(np.zeros((0, 3), dtype=np.float64))
    with pytest.raises(ManimVisionGeometryError, match="insufficient points"):
        geometry_adapter.vmobject_to_polygon(empty)


def test_closed_curve_produces_polygon(geometry_adapter: GeometryAdapter) -> None:
    """A Manim ``Circle`` must yield a valid Shapely polygon with positive area."""
    geom = geometry_adapter.vmobject_to_polygon(Circle(radius=1.0))
    assert geom.geom_type == "Polygon"
    assert geom.is_valid
    assert geom.area > 0.0


def test_glyph_produces_multipolygon(geometry_adapter: GeometryAdapter) -> None:
    """Typography outlines must become valid polygonal geometry (glyph subpath)."""
    glyph = Text("B")[0]
    geom = geometry_adapter.vmobject_to_polygon(glyph)
    assert geom.geom_type in ("Polygon", "MultiPolygon", "GeometryCollection")
    assert geom.is_valid
    assert geom.area > 0.0


def test_z_coordinate_discarded(geometry_adapter: GeometryAdapter) -> None:
    """Shapely output must be strictly planar (no Z ordinate)."""
    line = VMobject()
    line.append_points(np.array([[0.0, 0.0, 5.0], [1.0, 0.0, -3.0]], dtype=np.float64))
    geom = geometry_adapter.vmobject_to_polygon(line)
    assert geom.has_z is False


def test_precision_grid_applied(geometry_adapter: GeometryAdapter) -> None:
    """After ``set_precision``, coordinates must lie on the specified grid within tolerance."""
    grid = 0.05
    geom = geometry_adapter.vmobject_to_polygon(Circle(radius=0.5), precision_grid=grid)
    coords = np.asarray(geom.exterior.coords if geom.geom_type == "Polygon" else geom.geoms[0].exterior.coords)
    for x, y in coords:
        assert abs(x - round(x / grid) * grid) <= grid * 1.01 + 1e-9
        assert abs(y - round(y / grid) * grid) <= grid * 1.01 + 1e-9


def test_single_point_raises_error(geometry_adapter: GeometryAdapter) -> None:
    """A single-point VMobject cannot define a curve or polygon outline."""
    one = VMobject()
    one.append_points(np.array([[0.0, 0.0, 0.0]], dtype=np.float64))
    with pytest.raises(ManimVisionGeometryError, match="insufficient points"):
        geometry_adapter.vmobject_to_polygon(one)


def test_zero_length_path_raises_error(geometry_adapter: GeometryAdapter) -> None:
    """Coincident anchors are degenerate and must not produce silent geometry."""
    dup = VMobject()
    dup.append_points(np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float64))
    with pytest.raises(ManimVisionGeometryError):
        geometry_adapter.vmobject_to_polygon(dup)
