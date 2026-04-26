# Manim Vision

**Manim Vision** adds spatial collision awareness to [Manim Community](https://www.manim.community/) scenes. It watches your scene as `add`, `play`, and `remove` run, detects meaningful overlaps, and emits collision timelines that an LLM can use to fix layout problems without reading video frames.

|            | |
|------------|---|
| **PyPI**   | `pip install manim-vision` |
| **Python** | 3.10, 3.11, or 3.12 |
| **License**| MIT |

## What it does

- Hooks a live `Scene` without changing your source class at import time.
- Converts tracked `VMobject` geometry to Shapely shapes and checks overlap with an `STRtree`.
- Suppresses obvious noise such as glyph-on-glyph kerning, centered text-in-cell layouts, and tiny dust overlaps.
- Tracks overlap *intervals*, so a collision that persists for 100 plays is reported once with a start time and end time.
- Labels collisions with scene-meaningful names instead of raw memory ids.
- Produces a compact scene summary by default, with optional human-readable or silent modes.
- Can replay any monitored scene as a separate collision-overlay video for visual debugging.

## Installation

```bash
pip install manim-vision
```

Dependencies are declared in `pyproject.toml` and include `manim`, `shapely`, `wrapt`, `numpy`, and `jsonschema`.

## Quick start

Call `ManimVision.monitor(self)` early in `construct()`, then shut it down at the end so queued collision work can flush.

```python
from manim import BLUE, RED, Create, Circle, RIGHT, Scene, Square
from manim_vision import ManimVision


class MyScene(Scene):
    def construct(self):
        ManimVision.monitor(self)  # default: compact LLM summary

        circle = Circle(radius=1, color=BLUE)
        square = Square(side_length=1.2, color=RED)
        square.next_to(circle, RIGHT, buff=0)

        self.add(circle, square)
        self.play(Create(circle), Create(square))

        ManimVision.shutdown(self)
        print(ManimVision.results(self))
```

## Output modes

`ManimVision.monitor(scene, output_mode=...)` accepts one optional mode:

- `"llm"`: default. Collects collision events during the scene and writes one JSON timeline at shutdown to `media/manim_vision/<SceneName>_check_digest.jsonl`.
- `"human"`: collects the same events but writes a readable interval report for developers.
- `"silent"`: writes nothing to disk or stdout and keeps results available through `ManimVision.results(scene)`.

Example:

```python
ManimVision.monitor(self, output_mode="silent")
# ...
ManimVision.shutdown(self)
summary = ManimVision.results(self)
```

To generate a diagnostic overlay video after a run:

```python
overlay_path = ManimVision.render_overlay(
    r"C:\path\to\scene.py",
    "MyScene",
)
print(overlay_path)
```

## Output files

By default, Manim Vision writes under your Manim media directory:

- `media/manim_vision/<SceneName>_check_digest.jsonl`: JSON collision timeline in LLM mode.
- `media/manim_vision/<SceneName>_finalcontextcollisionreport.json`: compact LLM-facing report with interned object names and grouped event references.
- `media/manim_vision/<SceneName>_spatial_log.txt`: human-readable collision timeline in human mode.
- `media/manim_vision/<SceneName>_spatial.jsonl`: legacy per-event JSONL, only when `MANIM_VISION_PER_PAIR_JSONL=1`.

Override the directory with `MANIM_VISION_REPORT_DIR`.

## Public API

- `ManimVision.monitor(scene, output_mode="llm")`
- `ManimVision.shutdown(scene)`
- `ManimVision.results(scene)`
- `ManimVision.render_overlay(scene_or_script, scene_name=None, ...)`

Exceptions re-exported from `manim_vision`:

- `ManimVisionError`
- `ManimVisionGeometryError`
- `ManimVisionSchemaError`
- `ManimVisionProxyError`

## How it works

1. `ManimVision.monitor(scene)` attaches an engine, solver, dispatcher, lock, and worker executor to the live scene instance.
2. A mixin is inserted at runtime so `add`, `play`, and `remove` can register, resync, and deregister geometry.
3. Collision checks run on a single-worker background executor under a scene lock.
4. Raw collisions are filtered, semantically grouped, tracked as continuous overlap intervals, and flushed on shutdown.

## Notes

- Analysis is 2D and based on overlap area, not full 3D physics.
- Touching edges without positive overlap are ignored.
- Some concave cases use centroid-based fallbacks for fix hints.
- Internal lock-bearing state stays on proxies, so Manim creation-style animations still deep-copy safely.

## Development

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
python -m pytest
```

## License

MIT. See `LICENSE`.
