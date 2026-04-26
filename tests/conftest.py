"""Pytest fixtures for ManimVision tests."""

from __future__ import annotations

import pytest


@pytest.fixture
def geometry_adapter():
    """Provide a fresh :class:`~manim_vision.geometry.adapter.GeometryAdapter` instance."""
    from manim_vision.geometry.adapter import GeometryAdapter

    return GeometryAdapter()
