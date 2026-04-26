"""Tests for :mod:`manim_vision.solver.constraint`."""

from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon

from manim_vision.geometry.engine import CollisionResult
from manim_vision.solver.constraint import ConstraintSolver


def _cr_squares_overlap() -> CollisionResult:
    """Build a synthetic overlap between axis-aligned unit squares."""
    a = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    b = Polygon([(0.5, 0), (1.5, 0), (1.5, 1), (0.5, 1)])
    inter = a.intersection(b)
    return CollisionResult(
        mobject_a_id=1,
        mobject_b_id=2,
        mobject_a_name="A_1",
        mobject_b_name="B_2",
        geom_a=a,
        geom_b=b,
        overlap_area=float(inter.area),
        overlap_centroid=tuple(inter.centroid.coords[0]),
        overlap_geometry=inter,
    )


def test_mtv_pushes_apart() -> None:
    """SAT MTV for two convex squares must separate primarily along X by ~0.5 units."""
    solver = ConstraintSolver()
    mtv = solver.calculate_mtv(_cr_squares_overlap())
    assert mtv.shape == (3,)
    assert abs(mtv[0] - 0.5) <= 0.05 or abs(mtv[0] + 0.5) <= 0.05
    assert abs(mtv[1]) <= 0.05


def test_fix_syntax_up() -> None:
    """Positive ``dy`` must prefer ``shift(UP * magnitude)`` with four decimal places."""
    solver = ConstraintSolver()
    out = solver.generate_fix_syntax(np.array([0.0, 2.45, 0.0]))
    assert out == "shift(UP * 2.4500)"


def test_fix_syntax_left() -> None:
    """Negative ``dx`` must emit ``shift(LEFT * magnitude)`` after vertical component."""
    solver = ConstraintSolver()
    out = solver.generate_fix_syntax(np.array([-1.0, 0.0, 0.0]))
    assert out == "shift(LEFT * 1.0000)"


def test_fix_syntax_zero_displacement() -> None:
    """A null MTV must yield the documented no-op comment string."""
    solver = ConstraintSolver()
    out = solver.generate_fix_syntax(np.zeros(3))
    assert out == "# No significant displacement required"


def test_force_relaxation_convergence() -> None:
    """Five overlapping discs must accumulate non-zero displacements within the iteration cap."""
    from manim import Circle

    from manim_vision.geometry.engine import PrecisionGeometryEngine

    eng = PrecisionGeometryEngine()
    circles = [Circle(radius=1.2).shift([i * 0.08, (i % 2) * 0.05, 0.0]) for i in range(5)]
    for c in circles:
        eng.register(c)
    results = eng.check_collisions()
    solver = ConstraintSolver()
    disp = solver.apply_force_relaxation(results, iterations=50)
    assert len(disp) == 5
    for _mid, vec in disp.items():
        assert np.linalg.norm(vec) > 1e-6


def test_concave_shape_fallback() -> None:
    """Concave outlines must still return an MTV via the centroid fallback path."""
    concave = Polygon([(0, 0), (4, 0), (2, 0.8), (4, 4), (2, 3.0), (0, 4), (2, 2.0), (0, 0)])
    square = Polygon([(1.5, 1.0), (3.5, 1.0), (3.5, 3.0), (1.5, 3.0)])
    inter = concave.intersection(square)
    cr = CollisionResult(
        mobject_a_id=3,
        mobject_b_id=4,
        mobject_a_name="C_3",
        mobject_b_name="D_4",
        geom_a=concave,
        geom_b=square,
        overlap_area=float(inter.area),
        overlap_centroid=tuple(inter.centroid.coords[0]),
        overlap_geometry=inter,
    )
    solver = ConstraintSolver()
    mtv = solver.calculate_mtv(cr)
    assert np.all(np.isfinite(mtv))
    assert mtv.shape == (3,)
