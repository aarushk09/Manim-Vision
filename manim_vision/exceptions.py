"""Custom exception hierarchy for the Manim Vision library."""

from __future__ import annotations


class ManimVisionError(Exception):
    """Base exception for all Manim Vision library errors."""


class ManimVisionGeometryError(ManimVisionError):
    """Raised when a VMobject cannot be converted to a valid Shapely geometry."""


class ManimVisionSchemaError(ManimVisionError):
    """Raised when a generated telemetry payload fails JSON Schema validation."""


class ManimVisionProxyError(ManimVisionError):
    """Raised when the proxy instrumentation layer fails to wrap an object."""
