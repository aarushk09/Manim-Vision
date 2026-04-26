"""Build, validate, and emit JSON spatial health reports."""

from __future__ import annotations

import datetime
import json
import logging
import re
import sys
from typing import Any, TextIO

from jsonschema import ValidationError, validate

from manim_vision.exceptions import ManimVisionSchemaError
from manim_vision.telemetry.schema import MANIM_VISION_SPATIAL_REPORT_SCHEMA

logger = logging.getLogger("manim_vision.telemetry")


class TelemetryDispatcher:
    """Serializes collision telemetry to JSON after schema validation."""

    def __init__(self, output_stream: TextIO | None = None, scene_name: str = "UnknownScene") -> None:
        """Create a dispatcher writing to ``output_stream`` (default: stdout).

        Args:
            output_stream: Writable text stream for JSON lines.
            scene_name: Default Manim scene class name embedded in payloads.
        """
        self._stream: TextIO = output_stream or sys.stdout
        self._logger = logging.getLogger("manim_vision.telemetry")
        self._default_scene_name = scene_name

    def dispatch(
        self,
        collision_result: Any,
        mtv: Any,
        fix_syntax: str,
        *,
        scene_name: str | None = None,
    ) -> dict[str, Any]:
        """Build, validate, and dispatch a Spatial Health Report.

        Args:
            collision_result: A :class:`~manim_vision.geometry.engine.CollisionResult` instance.
            mtv: Iterable MTV components (length at least two).
            fix_syntax: Manim ``shift`` chain or comment string.
            scene_name: Optional override for the reporting scene name.

        Returns:
            The validated payload dictionary.

        Raises:
            ManimVisionSchemaError: If the payload fails JSON Schema validation.
        """
        name = scene_name or self._default_scene_name
        if isinstance(name, str):
            name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
        if not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z0-9_]+", name):
            self._logger.warning("Sanitizing scene name for schema pattern compliance.")
            name = "UnknownScene"

        payload = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
            "scene_name": name,
            "error_type": "OVERLAP",
            "colliding_entities": [
                collision_result.mobject_a_name,
                collision_result.mobject_b_name,
            ],
            "overlap_area": float(collision_result.overlap_area),
            "resolution_mtv": {
                "x": float(mtv[0]),
                "y": float(mtv[1]),
                "z": float(mtv[2]) if len(mtv) > 2 else 0.0,
            },
            "fix_suggestion": fix_syntax,
        }

        try:
            validate(instance=payload, schema=MANIM_VISION_SPATIAL_REPORT_SCHEMA)
        except ValidationError as exc:
            raise ManimVisionSchemaError(
                f"Generated telemetry payload failed schema validation: {exc.message}"
            ) from exc

        serialized = json.dumps(payload, indent=2)
        self._stream.write(serialized + "\n")
        self._stream.flush()
        return payload
