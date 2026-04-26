# Manim Vision

**Manim Vision** adds production-oriented **2D spatial collision awareness** to [Manim Community](https://www.manim.community/) scenes. It tracks `VMobject` geometry in the background, detects overlaps with [Shapely](https://shapely.readthedocs.io/) and a **STRtree** broad phase, and emits **JSON “spatial health” reports** (with suggested `shift(...)` fix strings) to help you find unintentional object overlap while you build animations.

|            | |
|------------|---|
| **PyPI**   | `pip install manim-vision` |
| **Python** | 3.10, 3.11, or 3.12 |
| **License**| MIT |

---

## Table of contents

- [What it does](#what-it-does)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Shutting down cleanly](#shutting-down-cleanly)
- [How it works (architecture)](#how-it-works-architecture)
- [Telemetry output](#telemetry-output)
- [Public API and exceptions](#public-api-and-exceptions)
- [Design notes and limitations](#design-notes-and-limitations)
- [Development](#development)
- [License](#license)

---

## What it does

- **Hooks** your live `Scene` so **`add`**, **`play`**, and **`remove`** participate in a collision pipeline (via a dynamically merged mixin; your scene class is unchanged at import time).
- **Wraps** added `VMobject`s with a transparent proxy that refreshes stored geometry when you call common **spatial** methods (`shift`, `scale`, `rotate`, `move_to`, `next_to`, `align_to`, `set_x` / `set_y` / `set_z`, `stretch`, `apply_matrix`, `apply_function`).
- **Detects** pairwise overlaps using **Shapely** (narrow phase, with DE-9IM / intersection area) on top of a **Shapely STRtree** for broad phase.
- **Relaxes** overlapping pairs with an internal **force-based iteration** and computes **MTV-style** separation hints (`ConstraintSolver`).
- **Emits** one **validated JSON** document per collision to **stdout** by default (JSON Schema: `MANIM_VISION_SPATIAL_REPORT_SCHEMA` in `manim_vision.telemetry.schema`).

Collision work runs on a **dedicated single-worker thread pool** so the main animation thread is not blocked by geometry queries.

---

## Installation

```bash
pip install manim-vision
```

Dependencies are declared in `pyproject.toml` and include at least: `manim`, `shapely`, `wrapt`, `numpy`, and `jsonschema`.

---

## Quick start

Call `ManimVision.monitor(self)` **early** in `construct()` (after you have a real `Scene` instance). Use your scene as usual: `add`, `play`, and `remove` are instrumented automatically.

```python
from manim import BLUE, RED, Create, Circle, Scene, Square
from manim_vision import ManimVision


class MyScene(Scene):
    def construct(self):
        ManimVision.monitor(self)

        a = Circle(radius=1, color=BLUE)
        b = Square(side_length=1.2, color=RED)
        b.next_to(a, RIGHT, buff=0)  # may overlap — reported if so

        self.add(a, b)
        self.play(Create(a), Create(b))

        ManimVision.shutdown(self)  # recommended: see below
```

**Lazy import:** `manim_vision` loads lightweight symbols (`ManimVisionError`, etc.) immediately; `ManimVision` is loaded from `manim_vision.core` the first time you access it, so imports stay cheap when Manim is not yet needed.

---

## Shutting down cleanly

`ManimVision.shutdown(self)` (or the mixin’s `shutdown()` on the scene) **waits for queued collision jobs** and **shuts down the worker thread pool** before the scene / process tears down. Use it at the end of `construct()` or after your last `add` / `play` in long or scripted runs to avoid pending work or pool warnings at exit.

---

## How it works (architecture)

1. **`ManimVision.monitor(scene)`** checks that `scene` is a Manim `Scene`, then attaches an engine, solver, telemetry dispatcher, **registry lock**, and **executor** to the instance. It rebases `scene.__class__` to a new type that **prepends** `ManimVisionSceneMixin` in the MRO so `add` / `play` / `remove` are overridden while the rest of your class behaves as before.
2. **`add`** wraps each `VMobject` in a **`ManimVisionMobjectProxy`**, which **registers** it with `PrecisionGeometryEngine` and **updates** geometry after spatial mutators.
3. After `add` and `play`, a collision check is **submitted** to the executor. Under the scene lock, the engine **checks collisions**, the solver **applies relaxation** and **MTV** math, and the **TelemetryDispatcher** writes **JSON** to the configured stream (default **stdout**).
4. **`remove`** deregisters mobjects from the engine’s registry.

Internal pieces (for reading the code or building on top of the library):

| Component | Role |
|-----------|------|
| `GeometryAdapter` | Converts `VMobject` outlines to Shapely geometry (with validation / error handling). |
| `PrecisionGeometryEngine` | Registry, `STRtree` queries, `CollisionResult` build-out. |
| `ConstraintSolver` | MTV / SAT-style hints, force relaxation, `shift(...)` fix string generation. |
| `TelemetryDispatcher` | Builds payloads, validates against JSON Schema, writes JSON lines. |

---

## Telemetry output

Each report is a **single JSON object** (pretty-printed with indentation in the default dispatcher) with fields such as:

- `timestamp` — ISO-8601 UTC
- `scene_name` — Sanitized scene class name
- `error_type` — e.g. `OVERLAP` (see schema for allowed values)
- `colliding_entities` — String labels `ClassName_id` for the two mobjects
- `overlap_area` — Positive float in world units²
- `resolution_mtv` — `x`, `y`, `z` components (2D analysis uses `z: 0`)
- `fix_suggestion` — Suggested Manim code, often chained `shift(UP * …).shift(RIGHT * …)` style

The contract is defined in code as `MANIM_VISION_SPATIAL_REPORT_SCHEMA` in `manim_vision/telemetry/schema.py`. Invalid payloads raise `ManimVisionSchemaError`.

To capture telemetry to a file when driving Manim headlessly, redirect **stdout** or construct a `TelemetryDispatcher(output_stream=...)` in advanced integrations (the default path used by the stock pipeline points at `sys.stdout`).

---

## Public API and exceptions

**Entry points**

- `manim_vision.ManimVision` — `monitor(scene)`, `shutdown(scene)`.

**Exceptions** (re-exported from `manim_vision`)

| Exception | Meaning |
|-----------|---------|
| `ManimVisionError` | Base error (e.g. `monitor()` given a non-`Scene`). |
| `ManimVisionGeometryError` | `VMobject` could not be turned into valid geometry. |
| `ManimVisionSchemaError` | Telemetry payload failed JSON Schema validation. |
| `ManimVisionProxyError` | Reserved for proxy/instrumentation failures. |

**Lower-level types** (used in tests and extensions) live under `manim_vision.geometry`, `manim_vision.proxy`, `manim_vision.solver`, and `manim_vision.telemetry` — e.g. `ManimVisionSceneProxy`, `PrecisionGeometryEngine`, `CollisionResult`, `ConstraintSolver`, `TelemetryDispatcher`.

---

## Design notes and limitations

- **2D-style analysis** in the engine: separation vectors use **xy**; mobjects are approximated for outline overlap in the scene plane. Not a full 3D physics engine.
- **VMobject-oriented**: tracking is built around vectorized mobjects; exotic object types may not register or may log conversion warnings.
- **Touches vs overlap**: pairs that only **touch** (zero area) are filtered out; positive **overlap area** is what triggers a report.
- **Heavier geometry** (e.g. concave shapes) may use **centroid / convex-hull fallbacks** in the solver; see `ConstraintSolver` docstrings and implementation for details.
- The library aims to be **non-invasive**: it does not replace your `self` reference; it only replaces the runtime class of the scene object to install hooks.
- **Manim’s `copy.deepcopy` and helpers** (used when building many animations) walk each mobject’s `__dict__`. Internals that hold a `threading.Lock` (e.g. the geometry engine) must **not** be stored on the underlying VMobject. Manim Vision keeps that state on the **wrapt proxy** and implements `__deepcopy__` on `ManimVisionMobjectProxy` / `ManimVisionSceneProxy` so creation-style animations (`FadeIn`, `Transform`, etc.) do not fail with `TypeError: cannot pickle '_thread.lock' object`.

---

## Development

From a checkout of the package root (the directory that contains `pyproject.toml`):

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -e ".[dev]"
python -m pytest
```

Run **Ruff** or your usual formatter if you keep them in the project; tests use **pytest** with `asyncio` mode as configured in `pyproject.toml`.

---

## License

MIT — see the `LICENSE` file in this repository.
