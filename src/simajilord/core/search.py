"""Small language-agnostic search helpers for local capability catalogs."""

from __future__ import annotations

import unicodedata


def normalize_search_text(value: str) -> str:
    """Apply NFKC/case folding and turn punctuation into token boundaries."""

    normalized = unicodedata.normalize("NFKC", value).casefold()
    characters = (
        character
        if character.isalnum()
        or unicodedata.category(character).startswith("M")
        else " "
        for character in normalized
    )
    return " ".join("".join(characters).split())


def search_features(value: str) -> frozenset[str]:
    """Return word/script features plus bounded CJK bigrams and trigrams."""

    features: set[str] = set()
    for token in normalize_search_text(value).split():
        features.add(token)
        for run, cjk in _script_runs(token):
            features.add(run)
            if not cjk:
                continue
            for size in (2, 3):
                features.update(
                    run[index : index + size]
                    for index in range(len(run) - size + 1)
                )
    return frozenset(features)


def search_overlap_score(query: str, candidate: str) -> int:
    """Score shared normalized features without language-specific stemming."""

    shared = search_features(query) & search_features(candidate)
    return min(24, sum(_feature_weight(feature) for feature in shared))


def phrase_match_score(query: str, phrases: tuple[str, ...]) -> int:
    """Reward descriptor-owned phrases, preferring specific multi-part intents."""

    normalized_query = normalize_search_text(query)
    if not normalized_query:
        return 0
    score = 0
    for phrase in phrases:
        normalized_phrase = normalize_search_text(phrase)
        if not normalized_phrase or (
            len(normalized_phrase) == 1
            and (
                not _is_cjk(normalized_phrase)
                or _is_hiragana(normalized_phrase)
            )
        ):
            continue
        if normalized_phrase in normalized_query:
            score += 2 + _phrase_specificity_bonus(normalized_phrase)
        elif len(normalized_query) >= 2 and normalized_query in normalized_phrase:
            score += 1 + (_phrase_specificity_bonus(normalized_phrase) // 2)
    return min(score, 24)


def normalized_substring(query: str, candidate: str, *, minimum: int = 3) -> bool:
    normalized_query = normalize_search_text(query)
    return (
        len(normalized_query) >= minimum
        and normalized_query in normalize_search_text(candidate)
    )


def _script_runs(value: str) -> tuple[tuple[str, bool], ...]:
    if not value:
        return ()
    runs: list[tuple[str, bool]] = []
    start = 0
    current_cjk = _is_cjk(value[0])
    for index, character in enumerate(value[1:], start=1):
        is_cjk = _is_cjk(character)
        if is_cjk == current_cjk:
            continue
        runs.append((value[start:index], current_cjk))
        start = index
        current_cjk = is_cjk
    runs.append((value[start:], current_cjk))
    return tuple(runs)


def _feature_weight(feature: str) -> int:
    if len(feature) < 2:
        return 0
    if all(_is_cjk(character) for character in feature):
        if all(_is_hiragana(character) for character in feature):
            return 0
        if len(feature) == 2:
            return 1
        if len(feature) == 3:
            return 2
        return 3
    return 2 if len(feature) >= 4 else 1


def _phrase_specificity_bonus(normalized_phrase: str) -> int:
    """Give a concrete action phrase more evidence than several generic nouns."""

    tokens = normalized_phrase.split()
    if len(tokens) > 1:
        return min(4, len(tokens))
    if any(_is_cjk(character) for character in normalized_phrase) and any(
        marker in normalized_phrase
        for marker in ("を", "に", "へ", "で", "から", "して", "する", "って", "一覧", "よう")
    ):
        return 3 if len(normalized_phrase) >= 8 else 2
    return 0


def _is_cjk(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x3040 <= codepoint <= 0x30FF
        or 0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xAC00 <= codepoint <= 0xD7AF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x20000 <= codepoint <= 0x2FA1F
    )


def _is_hiragana(character: str) -> bool:
    return 0x3040 <= ord(character) <= 0x309F
