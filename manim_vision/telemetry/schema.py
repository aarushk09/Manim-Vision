"""JSON Schema contract for Manim Vision spatial health telemetry."""

MANIM_VISION_SPATIAL_REPORT_SCHEMA: dict = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "ManimVisionSpatialHealthReport",
    "type": "object",
    "required": [
        "timestamp",
        "scene_name",
        "error_type",
        "colliding_entities",
        "overlap_area",
        "resolution_mtv",
        "fix_suggestion",
    ],
    "additionalProperties": False,
    "properties": {
        "timestamp": {
            "type": "string",
            "format": "date-time",
            "description": "ISO-8601 timestamp of collision detection event.",
        },
        "scene_name": {
            "type": "string",
            "pattern": "^[a-zA-Z0-9_]+$",
            "description": "The class name of the Manim Scene being monitored.",
        },
        "error_type": {
            "type": "string",
            "enum": ["OVERLAP", "OUT_OF_FRAME", "Z_INDEX_CONFLICT"],
            "description": "The categorical type of spatial violation.",
        },
        "colliding_entities": {
            "type": "array",
            "minItems": 2,
            "items": {"type": "string"},
            "description": "List of VMobject identifiers (class name + id) involved in the collision.",
        },
        "overlap_area": {
            "type": "number",
            "exclusiveMinimum": 0,
            "description": "The area of the intersecting geometry region in Manim world units squared.",
        },
        "resolution_mtv": {
            "type": "object",
            "required": ["x", "y", "z"],
            "additionalProperties": False,
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
                "z": {"type": "number"},
            },
            "description": "The raw Minimum Translation Vector components.",
        },
        "fix_suggestion": {
            "type": "string",
            "description": "A valid Python/Manim syntax string to resolve the collision.",
        },
    },
}
