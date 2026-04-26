"""Map low-level mobjects to human-scale labels and filter glyph-level noise."""

from __future__ import annotations

import os
from typing import Any

# Types whose sub-path overlaps are kerning / anti-alias noise, not layout bugs.
_TEXT_LIKE_NAMES: frozenset[str] = frozenset(
    {
        "Text",
        "MarkupText",
        "MathTex",
        "Tex",
        "SingleStringMathTex",
    }
)


def min_reportable_overlap_area() -> float:
    """Smallest world-unit² area for an overlap to be worth reporting (env override)."""
    raw = os.environ.get("MANIM_VISION_MIN_OVERLAP_AREA", "0.0001")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 0.0001


def _scene_root_for(mob: Any, scene: Any) -> Any | None:
    """The entry in ``scene.mobjects`` whose family contains ``mob`` (or ``mob`` is that root)."""
    for r in list(scene.mobjects):
        if mob is r or mob in r.get_family():
            return r
    return None


def _text_or_tex_ancestor(mob: Any, scene: Any) -> Any | None:
    """The innermost text-like mobject in the scene tree that still contains ``mob``."""
    root = _scene_root_for(mob, scene)
    if root is None:
        return None
    best: list[Any] = []
    for m in root.get_family():
        if type(m).__name__ not in _TEXT_LIKE_NAMES:
            continue
        if mob is m or mob in m.get_family():
            best.append(m)
    if not best:
        return None
    # Prefer the smallest sub-tree (typical: inner text block vs outer group).
    best.sort(key=lambda x: len(x.get_family()))
    return best[0]


def is_pair_internal_glyphs_same_text(mob_a: Any, mob_b: Any, scene: Any) -> bool:
    """True if both sides are different paths under the *same* Text/MathTeX (kerning, etc.)."""
    ta = _text_or_tex_ancestor(mob_a, scene)
    if ta is None:
        return False
    tb = _text_or_tex_ancestor(mob_b, scene)
    return ta is not None and ta is tb


def semantic_label(mob: Any, scene: Any) -> str:
    """Broad label: whole Text / Math string when possible, else a short root id."""
    anchor = _text_or_tex_ancestor(mob, scene)
    if anchor is not None:
        return _label_textish(anchor)
    root = _scene_root_for(mob, scene)
    if root is not None and root is not mob:
        cname = type(root).__name__
        return f"{cname}#{id(root) & 0xFFFF:04x}"
    return f"{type(mob).__name__}#{id(mob) & 0xFFFF:04x}"


def _label_textish(m: Any) -> str:
    cls = type(m).__name__
    if cls in ("Text", "MarkupText"):
        t = getattr(m, "text", None)
        if t is None:
            t = getattr(m, "original_text", None)
        if t is not None:
            s = str(t).replace("\n", " ").strip()
            if len(s) > 56:
                s = s[:53] + "…"
            return f'{cls}("{s}")'
    if cls in ("MathTex", "Tex", "SingleStringMathTex"):
        ts = getattr(m, "tex_string", None)
        if ts is None:
            ts = getattr(m, "tex", None)
        if ts is not None:
            t = str(ts).replace("\n", " ").strip()
            if len(t) > 56:
                t = t[:53] + "…"
            return f"{cls}({t!r})"
    return f"{cls}#{id(m) & 0xFFFF:04x}"


def stable_pair_key(label_a: str, label_b: str) -> str:
    a, b = sorted((label_a, label_b))
    return f"{a}↔{b}"


def session_dedupe_enabled() -> bool:
    """Session-level deduplication; disable with ``MANIM_VISION_DISABLE_SESSION_DEDUPE=1``."""
    return os.environ.get("MANIM_VISION_DISABLE_SESSION_DEDUPE", "").lower() not in (
        "1",
        "true",
        "yes",
    )


def per_pair_jsonl_enabled() -> bool:
    """Legacy: one JSON line per overlap pair. Default off (digest-only) — set ``MANIM_VISION_PER_PAIR_JSONL=1``."""
    return os.environ.get("MANIM_VISION_PER_PAIR_JSONL", "").lower() in (
        "1",
        "true",
        "yes",
    )


def is_strict_submobject(mob_ancestor: Any, mob_sub: Any) -> bool:
    """True if ``mob_sub`` is a strict descendant in Manim’s tree of ``mob_ancestor`` (kerning / label-in-tile / group outline)."""
    if mob_ancestor is None or mob_sub is None or mob_ancestor is mob_sub:
        return False
    try:
        if mob_sub in mob_ancestor.get_family():
            return True
    except (TypeError, AttributeError, RecursionError):
        return False
    return False


def is_intentional_layout_pair(mob_a: Any, mob_b: Any, scene: Any) -> bool:
    """Skip overlaps a human would *not* count as a mistake.

    1) **Nested mobject** — a label inside a :class:`VGroup`, a glyph inside :class:`Text`, etc. The outer drawn outline vs inner content is not a *cross* compositing bug.

    2) **Siblings in a 2-tile** — direct children ``Square``/``Circle``/``Rectangle``/``RoundedRectangle`` + :class:`Text` under the *same* small :class:`VGroup` (number-in-cell tiles).
    """
    if is_strict_submobject(mob_a, mob_b) or is_strict_submobject(mob_b, mob_a):
        return True
    return _sibling_shape_with_text_tiles(mob_a, mob_b, scene)


_SHAPE_TILE = frozenset({"Square", "Rectangle", "RoundedRectangle", "Circle", "Ellipse"})
_LABEL_IN_CELL = _TEXT_LIKE_NAMES


def _sibling_shape_with_text_tiles(mob_a: Any, mob_b: Any, scene: Any) -> bool:
    a_name, b_name = type(mob_a).__name__, type(mob_b).__name__
    for root in list(scene.mobjects):
        for g in root.get_family():
            if type(g).__name__ != "VGroup" or not g.submobjects:
                continue
            if mob_a in g.submobjects and mob_b in g.submobjects:
                st = {a_name, b_name}
                if st & _SHAPE_TILE and st & _LABEL_IN_CELL and len(g.submobjects) <= 6:
                    return True
    return False
