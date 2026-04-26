"""Map low-level VMobjects to human-readable collision participants and layout heuristics."""

from __future__ import annotations

import os
from collections import Counter
from typing import Any

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
    """Resolve tracked collision components into readable, stable scene-local labels."""

    def __init__(self, scene: Any) -> None:
        self._scene = scene
        self._family_members = self._collect_scene_family()
        self._ancestor_cache: dict[int, list[Any]] = {}
        self._owner_cache: dict[int, Any] = {}
        self._owner_base_cache: dict[int, str] = {}
        self._owner_label_cache: dict[int, str] = {}
        self._component_label_cache: dict[int, str] = {}
        self._component_members_cache: dict[int, list[Any]] = {}
        self._owners = self._collect_owners()
        self._owner_label_counts = Counter(self._owner_base_label(owner) for owner in self._owners)

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
        for child in list(getattr(member, "submobjects", ()) or ()):
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
        ancestors = [member for _size, member in matches]
        self._ancestor_cache[key] = ancestors
        return ancestors

    def _text_anchor(self, mob: Any) -> Any | None:
        for ancestor in self._ancestors(mob):
            if type(ancestor).__name__ in _TEXT_LIKE_NAMES:
                return ancestor
        return None

    def owner(self, mob: Any) -> Any:
        """Return the conceptual owner used for layout heuristics."""
        key = id(mob)
        cached = self._owner_cache.get(key)
        if cached is not None:
            return cached

        anchor = self._text_anchor(mob)
        if anchor is not None:
            owner = anchor
        else:
            owner = mob
            for ancestor in self._ancestors(mob):
                if type(ancestor).__name__ not in _GENERIC_LEAF_NAMES:
                    owner = ancestor
                    break
            if type(owner).__name__ in _GENERIC_LEAF_NAMES:
                scene_root = _scene_root_for(mob, self._scene)
                if scene_root is not None and type(scene_root).__name__ not in _GENERIC_LEAF_NAMES:
                    owner = scene_root
        self._owner_cache[key] = owner
        return owner

    def _owner_base_label(self, owner: Any) -> str:
        key = id(owner)
        cached = self._owner_base_cache.get(key)
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
                if context is not None:
                    label = f"{cls} for {context}"
                elif cls in _GENERIC_LEAF_NAMES:
                    nearby = self._nearest_context_label(owner)
                    if nearby is not None:
                        label = f"Path near {nearby}"
                    else:
                        center = getattr(owner, "get_center", lambda: (0.0, 0.0, 0.0))()
                        label = f"Path@({float(center[0]):.2f},{float(center[1]):.2f})"
                else:
                    label = cls

        self._owner_base_cache[key] = label
        return label

    def _owner_label(self, owner: Any) -> str:
        key = id(owner)
        cached = self._owner_label_cache.get(key)
        if cached is not None:
            return cached

        base = self._owner_base_label(owner)
        if self._owner_label_counts[base] <= 1:
            label = base
        else:
            peers = [peer for peer in self._owners if self._owner_base_label(peer) == base]
            ordinal = next(i for i, peer in enumerate(peers, start=1) if peer is owner)
            label = f"{base}[{ordinal}]"

        self._owner_label_cache[key] = label
        return label

    def _component_members(self, owner: Any) -> list[Any]:
        key = id(owner)
        cached = self._component_members_cache.get(key)
        if cached is not None:
            return cached

        members: list[Any] = []
        seen: set[int] = set()
        for member in self._family_members:
            if self.owner(member) is not owner:
                continue
            if not _is_component_leaf(member):
                continue
            member_key = id(member)
            if member_key in seen:
                continue
            seen.add(member_key)
            members.append(member)
        if not members:
            members = [owner]
        self._component_members_cache[key] = members
        return members

    def _component_index(self, owner: Any, mob: Any) -> int:
        members = self._component_members(owner)
        return next(index for index, member in enumerate(members) if member is mob)

    def _component_role(self, owner: Any) -> str:
        name = type(owner).__name__
        if name in {"Text", "MarkupText"}:
            return "char"
        if name in {"MathTex", "Tex", "SingleStringMathTex"}:
            return "glyph"
        return "part"

    def label(self, mob: Any) -> str:
        """Return a unique, readable label for the tracked collision component ``mob``."""
        key = id(mob)
        cached = self._component_label_cache.get(key)
        if cached is not None:
            return cached

        owner = self.owner(mob)
        owner_label = self._owner_label(owner)
        members = self._component_members(owner)
        if len(members) == 1 and members[0] is owner:
            label = owner_label
        elif mob in members:
            label = f"{owner_label}.{self._component_role(owner)}[{self._component_index(owner, mob)}]"
        else:
            label = owner_label

        self._component_label_cache[key] = label
        return label

    def pair_labels(self, mob_a: Any, mob_b: Any) -> tuple[str, str]:
        """Return a deterministic ordered label pair for human and machine output."""
        return tuple(sorted((self.label(mob_a), self.label(mob_b))))

    def event_key(self, mob_a: Any, mob_b: Any) -> tuple[str, str]:
        """Stable event key based on component labels."""
        return self.pair_labels(mob_a, mob_b)

    def is_pair_internal_glyphs_same_text(self, mob_a: Any, mob_b: Any) -> bool:
        """True if both sides are different paths under the same text-like object."""
        owner_a = self.owner(mob_a)
        owner_b = self.owner(mob_b)
        return owner_a is owner_b and type(owner_a).__name__ in _TEXT_LIKE_NAMES

    def is_intentional_layout_pair(self, mob_a: Any, mob_b: Any) -> bool:
        """Skip overlaps that are part of a composite object, not a real layout bug."""
        owner_a = self.owner(mob_a)
        owner_b = self.owner(mob_b)
        if _is_empty_text_like(owner_a) or _is_empty_text_like(owner_b):
            return True
        if is_strict_submobject(owner_a, owner_b) or is_strict_submobject(owner_b, owner_a):
            return True
        if _is_centered_text_in_shape(owner_a, owner_b) or _is_centered_text_in_shape(owner_b, owner_a):
            return True
        return _sibling_shape_with_text_tiles(owner_a, owner_b, self._scene)

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

    def _nearest_context_label(self, mob: Any) -> str | None:
        center_getter = getattr(mob, "get_center", None)
        if not callable(center_getter):
            return None
        center = center_getter()
        best_label: str | None = None
        best_distance: float | None = None
        for root in list(self._scene.mobjects):
            if root is mob:
                continue
            candidate_owner = self.owner(root)
            candidate_label = self._owner_base_label(candidate_owner)
            if candidate_label in {"VGroup", type(mob).__name__}:
                continue
            candidate_center_getter = getattr(candidate_owner, "get_center", None)
            if not callable(candidate_center_getter):
                continue
            candidate_center = candidate_center_getter()
            dx = float(candidate_center[0]) - float(center[0])
            dy = float(candidate_center[1]) - float(center[1])
            distance = (dx * dx) + (dy * dy)
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_label = candidate_label
        return best_label


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
    """Broad label: scene-local component label with text content when available."""
    return SceneSemanticResolver(scene).label(mob)


def _label_textish(mob: Any) -> str:
    cls = type(mob).__name__
    if cls in ("Text", "MarkupText"):
        text = getattr(mob, "original_text", None)
        if text is None:
            text = getattr(mob, "text", None)
        if text is not None:
            value = str(text).replace("\n", " ").strip()
            if len(value) > 56:
                value = value[:53] + "..."
            return f'{cls}("{value}")'
    if cls in ("MathTex", "Tex", "SingleStringMathTex"):
        tex = getattr(mob, "tex_string", None)
        if tex is None:
            tex = getattr(mob, "tex", None)
        if tex is not None:
            value = str(tex).replace("\n", " ").strip()
            if len(value) > 56:
                value = value[:53] + "..."
            return f"{cls}({value!r})"
    return cls


def _text_content(mob: Any) -> str:
    cls = type(mob).__name__
    if cls in ("Text", "MarkupText"):
        text = getattr(mob, "original_text", None)
        if text is None:
            text = getattr(mob, "text", None)
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
    """Return a deterministic key for an unordered pair of semantic labels."""
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
            if type(group).__name__ != "VGroup" or not getattr(group, "submobjects", None):
                continue
            if mob_a in group.submobjects and mob_b in group.submobjects:
                names = {a_name, b_name}
                if names & _SHAPE_TILE and names & _LABEL_IN_CELL and len(group.submobjects) <= 6:
                    return True
    return False


def _is_component_leaf(mob: Any) -> bool:
    children = list(getattr(mob, "submobjects", ()) or ())
    if children:
        return False
    points = getattr(mob, "points", None)
    try:
        return points is not None and len(points) >= 2
    except TypeError:
        return False
