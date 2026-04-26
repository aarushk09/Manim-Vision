"""Tests for :mod:`manim_vision.overlay`."""

from __future__ import annotations

from pathlib import Path

from shapely.geometry import Polygon

from manim_vision.overlay import (
    CollisionOverlayController,
    render_overlay_subprocess,
    serialize_overlap_geometry,
)


def test_overlay_group_contains_one_highlight_per_active_collision() -> None:
    """Active collisions at the same timestamp should each contribute one overlay polygon."""
    controller = CollisionOverlayController(
        scene_name="OverlayScene",
        collision_events=[
            {
                "objects": ["A", "B"],
                "start_time": 1.0,
                "end_time": 3.0,
                "samples": [
                    {
                        "time": 1.0,
                        "geometry": {"polygons": [{"shell": [[0, 0], [1, 0], [1, 1], [0, 0]], "holes": []}]},
                    }
                ],
            },
            {
                "objects": ["C", "D"],
                "start_time": 2.0,
                "end_time": 4.0,
                "samples": [
                    {
                        "time": 2.0,
                        "geometry": {"polygons": [{"shell": [[2, 2], [3, 2], [3, 3], [2, 2]], "holes": []}]},
                    }
                ],
            },
        ],
    )
    group = controller.overlay_group_at_time(2.5)
    assert len(group.submobjects) == 2


def test_overlay_timing_uses_interval_bounds() -> None:
    """Highlights should appear only while the collision interval is active."""
    controller = CollisionOverlayController(
        scene_name="OverlayScene",
        collision_events=[
            {
                "objects": ["A", "B"],
                "start_time": 1.0,
                "end_time": 2.0,
                "samples": [
                    {
                        "time": 1.0,
                        "geometry": {"polygons": [{"shell": [[0, 0], [1, 0], [1, 1], [0, 0]], "holes": []}]},
                    }
                ],
            }
        ],
    )
    assert len(controller.overlay_group_at_time(0.5).submobjects) == 0
    assert len(controller.overlay_group_at_time(1.5).submobjects) == 1
    assert len(controller.overlay_group_at_time(2.5).submobjects) == 0


def test_zero_duration_events_stay_visible_for_one_frame() -> None:
    """Snapshot-only collisions should still flash for a single rendered frame."""
    controller = CollisionOverlayController(
        scene_name="OverlayScene",
        collision_events=[
            {
                "objects": ["A", "B"],
                "start_time": 1.0,
                "end_time": 1.0,
                "samples": [
                    {
                        "time": 1.0,
                        "geometry": {"polygons": [{"shell": [[0, 0], [1, 0], [1, 1], [0, 0]], "holes": []}]},
                    }
                ],
            }
        ],
        instant_visibility_seconds=1 / 10,
    )
    assert len(controller.overlay_group_at_time(1.05).submobjects) == 1
    assert len(controller.overlay_group_at_time(1.2).submobjects) == 0


def test_serialize_overlap_geometry_preserves_polygon_shape() -> None:
    """Serialized overlap geometry should keep polygon coordinate data for replay."""
    geometry = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    payload = serialize_overlap_geometry(geometry)
    assert len(payload["polygons"]) == 1
    assert payload["polygons"][0]["shell"][0] == [0.0, 0.0]


def test_install_prefers_foreground_layer() -> None:
    """Overlay layers should attach in the foreground so text collisions stay visible."""

    class FakeScene:
        def __init__(self) -> None:
            self.foreground: list[object] = []
            self.added: list[object] = []
            self.renderer = type("Renderer", (), {"time": 1.0})()

        def add_foreground_mobject(self, mob: object) -> object:
            self.foreground.append(mob)
            return mob

        def add(self, mob: object) -> object:
            self.added.append(mob)
            return mob

    controller = CollisionOverlayController(scene_name="OverlayScene", collision_events=[])
    scene = FakeScene()
    layer = controller.install(scene)
    assert scene.foreground == [layer]
    assert scene.added == []


def test_render_overlay_subprocess_uses_script_directory(monkeypatch) -> None:
    """Overlay re-renders should land beside the original script's media output."""
    recorded: dict[str, object] = {}

    def fake_run(cmd, *, check, env, cwd):  # type: ignore[no-untyped-def]
        recorded["cmd"] = cmd
        recorded["check"] = check
        recorded["env"] = env
        recorded["cwd"] = cwd

    monkeypatch.setattr("manim_vision.overlay.subprocess.run", fake_run)

    render_overlay_subprocess(
        script_path=Path(r"C:\tmp\demo\scene.py"),
        scene_name="DemoScene",
        report_path=Path(r"C:\tmp\demo\media\manim_vision\DemoScene_check_digest.jsonl"),
        output_file="DemoScene_collision_overlay",
    )

    assert recorded["cwd"] == r"C:\tmp\demo"
