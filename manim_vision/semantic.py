"""Map low-level mobjects to meaningful scene entities and suppress layout noise."""

from __future__ import annotations

import os
from collections import Counter
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

_SHAPE_TILE = frozenset({"Square", "Rectangle", "RoundedRectangle", "Circle", "Ellipse"})
_LABEL_IN_CELL = _TEXT_LIKE_NAMES
_GENERIC_LEAF_NAMES = frozenset({"VMobjectFromSVGPath", "VectorizedPoint"})


def min_reportable_overlap_area() -> float:
    """Smallest world-unit^2 area for an overlap to be worth reporting (env override)."""
    raw = os.environ.get("MANIM_VISION_MIN_OVERLAP_AREA", "0.0001")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 0.0001


class SceneSemanticResolver:
    """Resolve raw mobjects into stable, human-readable scene entities.

    Collision checks happen on leaf SVG paths and group wrappers, but an LLM needs
    names for the *conceptual* objects involved, not a stream of glyph path ids.
    """

    def __init__(self, scene: Any) -> None:
        self._scene = scene
        self._family_members = self._collect_scene_family()
        self._ancestor_cache: dict[int, list[Any]] = {}
        self._owner_cache: dict[int, Any] = {}
        self._base_label_cache: dict[int, str] = {}
        self._label_cache: dict[int, str] = {}
        self._owners = self._collect_owners()
        self._label_counts = Counter(self._base_label(owner) for owner in self._owners)

    def _collect_scene_family(self) -> list[Any]:
        seen: set[int] = set()
        ordered: list[Any] = []
        for root in list(self._scene.mobjects):
            self._walk_member(root, seen, ordered)
        return ordered

    def _walk_member(self, member: Any, seen: set[int], ordered: list[Any]) -> None:
        key = id(member)
        if key in seen:
            return
        seen.add(key)
        ordered.append(member)

        family_getter = getattr(member, "get_family", None)
        if callable(family_getter):
            for relative in family_getter():
                rel_key = id(relative)
                if rel_key in seen:
                    continue
                seen.add(rel_key)
                ordered.append(relative)
                for child in getattr(relative, "submobjects", ()) or ():
                    self._walk_member(child, seen, ordered)

        for child in getattr(member, "submobjects", ()) or ():
            self._walk_member(child, seen, ordered)

    def _collect_owners(self) -> list[Any]:
        seen: set[int] = set()
        owners: list[Any] = []
        for member in self._family_members:
            owner = self.owner(member)
            key = id(owner)
            if key in seen:
                continue
            seen.add(key)
            owners.append(owner)
        return owners

    def _ancestors(self, mob: Any) -> list[Any]:
        key = id(mob)
        cached = self._ancestor_cache.get(key)
        if cached is not None:
            return cached

        matches: list[tuple[int, Any]] = []
        for member in self._family_members:
            family_getter = getattr(member, "get_family", None)
            family = family_getter() if callable(family_getter) else (member,)
            if mob is member or mob in family:
                matches.append((len(family), member))
        matches.sort(key=lambda item: item[0])
        out = [member for _size, member in matches]
        self._ancestor_cache[key] = out
        return out

    def _text_anchor(self, mob: Any) -> Any | None:
        for ancestor in self._ancestors(mob):
            if type(ancestor).__name__ in _TEXT_LIKE_NAMES:
                return ancestor
        return None

    def owner(self, mob: Any) -> Any:
        """Return the representative object for grouping and event lifecycle tracking."""
        key = id(mob)
        cached = self._owner_cache.get(key)
        if cached is not None:
            return cached
        text_anchor = self._text_anchor(mob)
        if text_anchor is not None:
            owner = text_anchor
        else:
            owner = mob
            for ancestor in self._ancestors(mob):
                if type(ancestor).__name__ not in _GENERIC_LEAF_NAMES:
                    owner = ancestor
                    break
        self._owner_cache[key] = owner
        return owner

    def _tile_context(self, mob: Any) -> str | None:
        for ancestor in self._ancestors(mob):
            if type(ancestor).__name__ != "VGroup":
                continue
            children = list(getattr(ancestor, "submobjects", ()) or ())
            if not children or len(children) > 6:
                continue
            if not any(
                child is mob or mob in getattr(child, "get_family", lambda: (child,))()
                for child in children
            ):
                continue
            text_children = [
                child for child in children if type(self.owner(child)).__name__ in _TEXT_LIKE_NAMES
            ]
            if not text_children:
                continue
            return _label_textish(self.owner(text_children[0]))
        return None

    def _base_label(self, mob: Any) -> str:
        owner = self.owner(mob)
        key = id(owner)
        cached = self._base_label_cache.get(key)
        if cached is not None:
            return cached

        if type(owner).__name__ in _TEXT_LIKE_NAMES:
            label = _label_textish(owner)
        else:
            given_name = str(getattr(owner, "name", "") or "").strip()
            if given_name and given_name != type(owner).__name__:
                label = given_name
            else:
                context = self._tile_context(owner)
                cls = type(owner).__name__
                label = f"{cls} for {context}" if context is not None else cls

        self._base_label_cache[key] = label
        return label

    def label(self, mob: Any) -> str:
        """Return a unique, human-readable label for ``mob`` within the current scene."""
        owner = self.owner(mob)
        key = id(owner)
        cached = self._label_cache.get(key)
        if cached is not None:
            return cached

        base = self._base_label(owner)
        if self._label_counts[base] <= 1:
            label = base
        else:
            peers = [peer for peer in self._owners if self._base_label(peer) == base]
            ordinal = next(i for i, peer in enumerate(peers, start=1) if peer is owner)
            label = f"{base}[{ordinal}]"

        self._label_cache[key] = label
        return label

    def pair_labels(self, mob_a: Any, mob_b: Any) -> tuple[str, str]:
        """Return a deterministic ordered label pair for human and LLM output."""
        return tuple(sorted((self.label(mob_a), self.label(mob_b))))

    def event_key(self, mob_a: Any, mob_b: Any) -> tuple[int, int]:
        """Stable event key based on semantic owner identities."""
        ids = sorted((id(self.owner(mob_a)), id(self.owner(mob_b))))
        return (ids[0], ids[1])

    def is_pair_internal_glyphs_same_text(self, mob_a: Any, mob_b: Any) -> bool:
        """True if both sides are different paths under the same text object."""
        ta = self._text_anchor(mob_a)
        if ta is None:
            return False
        tb = self._text_anchor(mob_b)
        return tb is ta

    def is_intentional_layout_pair(self, mob_a: Any, mob_b: Any) -> bool:
        """Skip overlaps that are part of a composite object, not a scene bug."""
        owner_a = self.owner(mob_a)
        owner_b = self.owner(mob_b)
        if _is_empty_text_like(owner_a) or _is_empty_text_like(owner_b):
            return True
        if is_strict_submobject(owner_a, owner_b) or is_strict_submobject(owner_b, owner_a):
            return True
        if _is_centered_text_in_shape(owner_a, owner_b) or _is_centered_text_in_shape(owner_b, owner_a):
            return True
        return _sibling_shape_with_text_tiles(owner_a, owner_b, self._scene)


def _scene_root_for(mob: Any, scene: Any) -> Any | None:
    """The entry in ``scene.mobjects`` whose family contains ``mob`` (or ``mob`` is that root)."""
    for root in list(scene.mobjects):
        if mob is root or mob in root.get_family():
            return root
    return None


def _text_or_tex_ancestor(mob: Any, scene: Any) -> Any | None:
    """Compatibility wrapper for the innermost text-like ancestor that contains ``mob``."""
    return SceneSemanticResolver(scene)._text_anchor(mob)


def is_pair_internal_glyphs_same_text(mob_a: Any, mob_b: Any, scene: Any) -> bool:
    """True if both sides are different paths under the same text object."""
    return SceneSemanticResolver(scene).is_pair_internal_glyphs_same_text(mob_a, mob_b)


def semantic_label(mob: Any, scene: Any) -> str:
    """Broad label: whole Text / Math string when possible, else a stable scene-local label."""
    return SceneSemanticResolver(scene).label(mob)


def _label_textish(m: Any) -> str:
    cls = type(m).__name__
    if cls in ("Text", "MarkupText"):
        text = getattr(m, "text", None)
        if text is None:
            text = getattr(m, "original_text", None)
        if text is not None:
            value = str(text).replace("\n", " ").strip()
            if len(value) > 56:
                value = value[:53] + "..."
            return f'{cls}("{value}")'
    if cls in ("MathTex", "Tex", "SingleStringMathTex"):
        tex = getattr(m, "tex_string", None)
        if tex is None:
            tex = getattr(m, "tex", None)
        if tex is not None:
            value = str(tex).replace("\n", " ").strip()
            if len(value) > 56:
                value = value[:53] + "..."
            return f"{cls}({value!r})"
    return cls


def _text_content(mob: Any) -> str:
    cls = type(mob).__name__
    if cls in ("Text", "MarkupText"):
        text = getattr(mob, "text", None)
        if text is None:
            text = getattr(mob, "original_text", None)
        return str(text or "")
    if cls in ("MathTex", "Tex", "SingleStringMathTex"):
        text = getattr(mob, "tex_string", None)
        if text is None:
            text = getattr(mob, "tex", None)
        return str(text or "")
    return ""


def _is_empty_text_like(mob: Any) -> bool:
    return type(mob).__name__ in _TEXT_LIKE_NAMES and not _text_content(mob).strip()


def _is_centered_text_in_shape(shape: Any, text: Any) -> bool:
    if type(shape).__name__ not in _SHAPE_TILE or type(text).__name__ not in _TEXT_LIKE_NAMES:
        return False
    try:
        text_width = float(getattr(text, "width", 0.0))
        text_height = float(getattr(text, "height", 0.0))
        shape_width = float(getattr(shape, "width", 0.0))
        shape_height = float(getattr(shape, "height", 0.0))
        if min(shape_width, shape_height) <= 0.0:
            return False
        center_delta = shape.get_center() - text.get_center()
        max_center_offset = min(shape_width, shape_height) * 0.18
        fits_inside = text_width <= shape_width * 0.92 and text_height <= shape_height * 0.92
        centered = abs(float(center_delta[0])) <= max_center_offset and abs(float(center_delta[1])) <= max_center_offset
        return fits_inside and centered
    except Exception:
        return False


def stable_pair_key(label_a: str, label_b: str) -> str:
    a, b = sorted((label_a, label_b))
    return f"{a}<->{b}"


def session_dedupe_enabled() -> bool:
    """Session-level deduplication; disable with ``MANIM_VISION_DISABLE_SESSION_DEDUPE=1``."""
    return os.environ.get("MANIM_VISION_DISABLE_SESSION_DEDUPE", "").lower() not in (
        "1",
        "true",
        "yes",
    )


def per_pair_jsonl_enabled() -> bool:
    """Legacy per-pair JSONL mode; disabled by default."""
    return os.environ.get("MANIM_VISION_PER_PAIR_JSONL", "").lower() in (
        "1",
        "true",
        "yes",
    )


def is_strict_submobject(mob_ancestor: Any, mob_sub: Any) -> bool:
    """True if ``mob_sub`` is a strict descendant in Manim's tree of ``mob_ancestor``."""
    if mob_ancestor is None or mob_sub is None or mob_ancestor is mob_sub:
        return False
    try:
        if mob_sub in mob_ancestor.get_family():
            return True
    except (TypeError, AttributeError, RecursionError):
        return False
    return False


def is_intentional_layout_pair(mob_a: Any, mob_b: Any, scene: Any) -> bool:
    """Skip overlaps that a human would treat as designed composition, not a bug."""
    return SceneSemanticResolver(scene).is_intentional_layout_pair(mob_a, mob_b)


def _sibling_shape_with_text_tiles(mob_a: Any, mob_b: Any, scene: Any) -> bool:
    a_name, b_name = type(mob_a).__name__, type(mob_b).__name__
    for root in list(scene.mobjects):
        for group in root.get_family():
            if type(group).__name__ != "VGroup" or not group.submobjects:
                continue
            if mob_a in group.submobjects and mob_b in group.submobjects:
                st = {a_name, b_name}
                if st & _SHAPE_TILE and st & _LABEL_IN_CELL and len(group.submobjects) <= 6:
                    return True
    return False
