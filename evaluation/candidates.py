"""Candidate-set construction for recommendation evaluation."""

from __future__ import annotations

from collections.abc import Hashable, Iterable


def full_catalog_candidates(
    catalog_items: Iterable[Hashable],
    seen_items: Iterable[Hashable],
    relevant_items: Iterable[Hashable],
) -> tuple[Hashable, ...]:
    """Return leakage-safe full-catalog candidates.

    Items already present in the user's observed history are excluded.
    Relevant held-out items are always retained, even if the caller
    accidentally includes one in ``seen_items``.
    """
    catalog = set(catalog_items)
    seen = set(seen_items)
    relevant = set(relevant_items)

    missing = relevant.difference(catalog)
    if missing:
        raise ValueError(
            f"Relevant items missing from catalog: "
            f"{sorted(missing, key=repr)}"
        )

    candidates = (catalog - seen) | relevant
    return tuple(sorted(candidates, key=repr))
