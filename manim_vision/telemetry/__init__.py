"""Telemetry schema validation and dispatch."""

from __future__ import annotations

from manim_vision.telemetry.dispatcher import TelemetryDispatcher
from manim_vision.telemetry.schema import MANIM_VISION_SPATIAL_REPORT_SCHEMA

__all__ = ["MANIM_VISION_SPATIAL_REPORT_SCHEMA", "TelemetryDispatcher"]
