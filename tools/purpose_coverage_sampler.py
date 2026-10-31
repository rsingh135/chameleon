"""Deterministic TRAIN-record coverage within a class, independent of class policy.

The worker chooses labels with its existing balanced or frequency-based policy.
For each chosen label, this sampler visits every record before repeating one.
Each shuffled deck interleaves families, so its first F draws reach all F families.
This guarantees exposure, not label correctness or learned-model improvement.

Checkpoint ``state_dict()`` explicitly alongside the worker's other RNG state.
Restore with ``ClassCoverageSampler.from_state(records, state)`` or
``sampler.restore(state)``. State is JSON-compatible and dataset-bound; no pickle,
Torch, model, image, or network dependencies are used. Input payload/mask selection
belongs to the data owner; only record identity, class, family and split are read.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict

_KIND = "purpose_class_coverage_sampler"
_STATE_KEYS = {
    "schema_version",
    "kind",
    "seed",
    "records_sha256",
    "record_count",
    "rng_state",
    "draw_count",
    "classes",
}
_CLASS_KEYS = {"deck", "cursor", "cycle", "draw_count", "exposures"}


def _integer(value, name, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _keys(value, expected, name):
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{name} has an unsupported field set")


def _rng_from_json(value):
    # random.Random.getstate() version 3: 624 uint32 words, index, no Gaussian cache.
    if (
        not isinstance(value, list)
        or len(value) != 3
        or type(value[0]) is not int
        or value[0] != 3
        or not isinstance(value[1], list)
        or len(value[1]) != 625
        or value[2] is not None
    ):
        raise ValueError("Invalid sampler RNG state")
    if any(type(v) is not int or not 0 <= v <= 0xFFFFFFFF for v in value[1][:-1]):
        raise ValueError("Invalid sampler RNG words")
    if type(value[1][-1]) is not int or not 0 <= value[1][-1] <= 624:
        raise ValueError("Invalid sampler RNG index")
    rng = random.Random()
    try:
        rng.setstate((3, tuple(value[1]), None))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Invalid sampler RNG state") from error
    return rng


class ClassCoverageSampler:
    """Draw record IDs without replacement within each independently chosen label.

    Every input row must declare ``split='train'`` and nonempty string
    ``record_id``, ``label`` and ``family_id``. IDs must be globally unique.
    Input order does not affect seeded behavior. No global RNG state is changed.
    """

    def __init__(self, records, *, seed):
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        identities = []
        seen = set()
        for row in records:
            if not isinstance(row, dict) or row.get("split") != "train":
                raise ValueError("Coverage sampler accepts TRAIN rows only")
            for field in ("record_id", "label", "family_id"):
                if not isinstance(row.get(field), str) or not row[field].strip():
                    raise ValueError(f"Each TRAIN row requires a nonempty {field}")
            if row["record_id"] in seen:
                raise ValueError("Duplicate record_id in TRAIN coverage records")
            seen.add(row["record_id"])
            identities.append({k: row[k] for k in ("record_id", "label", "family_id")})
        if not identities:
            raise ValueError("Coverage sampler requires at least one TRAIN record")
        identities.sort(key=lambda row: row["record_id"])
        frozen = {
            "protocol": "train-record-label-family-v1",
            "split": "train",
            "records": identities,
        }
        canonical = json.dumps(frozen, sort_keys=True, separators=(",", ":"), allow_nan=False)
        self.records_sha256 = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        self._seed = seed
        self._rng = random.Random(seed)
        self._families = {row["record_id"]: row["family_id"] for row in identities}
        self._pools = defaultdict(lambda: defaultdict(list))
        for row in identities:
            self._pools[row["label"]][row["family_id"]].append(row["record_id"])
        self._states = {}
        for label in sorted(self._pools):
            deck = self._new_deck(label)
            self._states[label] = {
                "deck": deck,
                "cursor": 0,
                "cycle": 0,
                "draw_count": 0,
                "exposures": dict.fromkeys(deck, 0),
            }
        self._draw_count = 0

    def _new_deck(self, label):
        family_order = sorted(self._pools[label])
        self._rng.shuffle(family_order)
        pools = {family: list(self._pools[label][family]) for family in family_order}
        for pool in pools.values():
            self._rng.shuffle(pool)
        return [
            pool[index]
            for index in range(max(map(len, pools.values())))
            for pool in pools.values()
            if index < len(pool)
        ]

    def draw(self, label):
        """Return the next TRAIN record_id for a label chosen by the caller."""
        if not isinstance(label, str) or label not in self._states:
            raise ValueError("Unknown TRAIN class label")
        state = self._states[label]
        if state["cursor"] == len(state["deck"]):
            state["deck"] = self._new_deck(label)
            state["cursor"] = 0
            state["cycle"] += 1
        record_id = state["deck"][state["cursor"]]
        state["cursor"] += 1
        state["draw_count"] += 1
        state["exposures"][record_id] += 1
        self._draw_count += 1
        return record_id

    def state_dict(self):
        """Return detached JSON-compatible state, including this sampler's private RNG."""
        rng = self._rng.getstate()
        return {
            "schema_version": 1,
            "kind": _KIND,
            "seed": self._seed,
            "records_sha256": self.records_sha256,
            "record_count": len(self._families),
            "rng_state": [rng[0], list(rng[1]), rng[2]],
            "draw_count": self._draw_count,
            "classes": {
                label: {**state, "deck": list(state["deck"]), "exposures": dict(state["exposures"])}
                for label, state in self._states.items()
            },
        }

    def _validate_deck(self, label, deck):
        expected = {record_id for pool in self._pools[label].values() for record_id in pool}
        if (
            not isinstance(deck, list)
            or any(not isinstance(r, str) for r in deck)
            or len(deck) != len(expected)
            or set(deck) != expected
        ):
            raise ValueError(f"Invalid deck permutation for {label}")
        family_order = [self._families[r] for r in deck[: len(self._pools[label])]]
        if set(family_order) != set(self._pools[label]):
            raise ValueError(f"Deck does not reach every family early for {label}")
        expected_order = [
            family
            for i in range(max(map(len, self._pools[label].values())))
            for family in family_order
            if i < len(self._pools[label][family])
        ]
        if [self._families[r] for r in deck] != expected_order:
            raise ValueError(f"Deck is not family-interleaved for {label}")

    def restore(self, state):
        """Validate the entire frozen dataset/deck/RNG state before changing this sampler."""
        _keys(state, _STATE_KEYS, "Sampler state")
        if (
            type(state["schema_version"]) is not int
            or state["schema_version"] != 1
            or state["kind"] != _KIND
            or type(state["seed"]) is not int
            or state["seed"] != self._seed
        ):
            raise ValueError("Unsupported sampler state identity or seed")
        if (
            state["records_sha256"] != self.records_sha256
            or type(state["record_count"]) is not int
            or state["record_count"] != len(self._families)
        ):
            raise ValueError("Frozen TRAIN dataset identity changed")
        _keys(state["classes"], set(self._states), "Sampler classes")
        rng = _rng_from_json(state["rng_state"])
        restored = {}
        total = 0
        for label, saved in state["classes"].items():
            _keys(saved, _CLASS_KEYS, f"Class state {label}")
            deck = saved["deck"]
            self._validate_deck(label, deck)
            n = len(deck)
            cursor = _integer(saved["cursor"], "cursor")
            cycle = _integer(saved["cycle"], "cycle")
            count = _integer(saved["draw_count"], "class draw_count")
            if cursor > n or count != cycle * n + cursor or (cycle > 0 and cursor == 0):
                raise ValueError(f"Inconsistent deck cursor/cycle/draw count for {label}")
            _keys(saved["exposures"], set(deck), f"Class exposures {label}")
            prefix = set(deck[:cursor])
            exposures = {}
            for record_id, value in saved["exposures"].items():
                value = _integer(value, "exposure count")
                if value != cycle + (record_id in prefix):
                    raise ValueError(f"Exposure counts disagree with deck cursor for {label}")
                exposures[record_id] = value
            restored[label] = {
                "deck": list(deck),
                "cursor": cursor,
                "cycle": cycle,
                "draw_count": count,
                "exposures": exposures,
            }
            total += count
        if _integer(state["draw_count"], "total draw_count") != total:
            raise ValueError("Total draw count disagrees with class counts")
        self._states, self._rng, self._draw_count = restored, rng, total
        return self

    @classmethod
    def from_state(cls, records, state):
        """Create a sampler for the same TRAIN records and restore validated JSON state."""
        _keys(state, _STATE_KEYS, "Sampler state")
        sampler = cls(records, seed=state["seed"])
        return sampler.restore(state)

    def coverage_report(self):
        """Observed exposure only; full first-pass coverage needs n draws per class."""
        classes = {}
        for label, state in self._states.items():
            counts = state["exposures"]
            seen = {record_id for record_id, count in counts.items() if count}
            missing = sorted(set(counts) - seen)
            classes[label] = {
                "records": len(counts),
                "unique_records_seen": len(seen),
                "families": len(self._pools[label]),
                "unique_families_seen": len({self._families[r] for r in seen}),
                "missing_record_ids": missing,
                "draw_count": state["draw_count"],
                "repeated_exposures": sum(max(0, count - 1) for count in counts.values()),
                "minimum_draws_for_full_class_coverage": len(counts),
                "minimum_additional_draws_for_first_coverage": len(missing),
                "record_exposures": dict(sorted(Counter(counts).items())),
            }
        return {
            "records_sha256": self.records_sha256,
            "draw_count": self._draw_count,
            "classes": classes,
        }
