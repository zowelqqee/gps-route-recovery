"""A random generator specified precisely enough to be reimplemented in Swift.

The baseline normally uses ``numpy.random.default_rng``, whose PCG64 bit stream
and ziggurat normal sampler cannot be reproduced in another language without
porting a large part of numpy. That is fine for the batch processor, but it
leaves the Swift on-device implementation comparable only through "the endpoints
are within a few metres".

``ParityGenerator`` is the alternative: PCG-XSH-RR 64/32 exactly as published,
with every derived distribution defined by its *consumption order*, because that
is the part that has to match. ``LiveTracking/DeterministicRNG.swift`` is a
line-for-line counterpart, so with ``--rng parity`` on both sides the particle
filter sees an identical stream of random numbers and the two implementations
can be compared far more tightly.

It exposes the subset of the ``numpy.random.Generator`` surface that
:mod:`geotrace.particle_filter` actually uses, so it can be dropped in without
touching the filter.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

_MULTIPLIER = 6364136223846793005
_MASK64 = (1 << 64) - 1
_MASK32 = (1 << 32) - 1
DEFAULT_SEQUENCE = 0xDA3E39CB94B95BDB


class ParityGenerator:
    """PCG32 with numpy-compatible method names."""

    def __init__(self, seed: int, sequence: int = DEFAULT_SEQUENCE) -> None:
        self._state = 0
        self._increment = ((sequence << 1) | 1) & _MASK64
        self._next_uint32()
        self._state = (self._state + int(seed)) & _MASK64
        self._next_uint32()

    # ------------------------------------------------------------- core

    def _next_uint32(self) -> int:
        old = self._state
        self._state = (old * _MULTIPLIER + self._increment) & _MASK64
        xorshifted = (((old >> 18) ^ old) >> 27) & _MASK32
        rot = (old >> 59) & 31
        return ((xorshifted >> rot) | (xorshifted << ((-rot) & 31))) & _MASK32

    def _next_uint64(self) -> int:
        return ((self._next_uint32() << 32) | self._next_uint32()) & _MASK64

    def _next_double(self) -> float:
        """Uniform in [0, 1) with 53 significant bits."""
        return (self._next_uint64() >> 11) * (1.0 / 9007199254740992.0)

    # ------------------------------------------------------------ bulk path
    #
    # The recurrence is sequential, but PCG's state advance is affine
    # (s_{i+1} = s_i * M + I), so advancing by i steps is itself affine and can
    # be composed by repeated doubling. That turns "generate n outputs" into
    # log2(n) vectorised passes and makes parity mode usable on a real trip:
    # a 5000-particle filter over an eight-minute drive needs tens of millions
    # of draws, which a scalar Python loop cannot deliver.
    #
    # `_next_uint32_bulk` is checked against the scalar path in the tests, so
    # the two can never produce different streams.

    def _next_uint32_bulk(self, count: int) -> np.ndarray:
        if count <= 0:
            return np.zeros(0, dtype=np.uint64)
        steps = np.arange(count + 1, dtype=np.uint64)
        multiplier = np.ones(count + 1, dtype=np.uint64)
        addend = np.zeros(count + 1, dtype=np.uint64)

        step_multiplier = np.uint64(_MULTIPLIER)
        step_addend = np.uint64(self._increment)
        bit = 0
        with np.errstate(over="ignore"):
            while (1 << bit) <= count:
                mask = ((steps >> np.uint64(bit)) & np.uint64(1)).astype(bool)
                addend[mask] = step_multiplier * addend[mask] + step_addend
                multiplier[mask] = multiplier[mask] * step_multiplier
                step_addend = step_multiplier * step_addend + step_addend
                step_multiplier = step_multiplier * step_multiplier
                bit += 1

            state = np.uint64(self._state)
            states = multiplier * state + addend
            old = states[:count]
            xorshifted = (((old >> np.uint64(18)) ^ old) >> np.uint64(27)) & np.uint64(_MASK32)
            rot = (old >> np.uint64(59)) & np.uint64(31)
            rotated = (
                (xorshifted >> rot) | (xorshifted << ((np.uint64(32) - rot) & np.uint64(31)))
            ) & np.uint64(_MASK32)
            self._state = int(states[count])
        return rotated

    def _next_double_bulk(self, count: int) -> np.ndarray:
        if count <= 0:
            return np.zeros(0)
        words = self._next_uint32_bulk(count * 2)
        high = words[0::2].astype(np.uint64)
        low = words[1::2].astype(np.uint64)
        combined = (high << np.uint64(32)) | low
        return (combined >> np.uint64(11)).astype(np.float64) * (1.0 / 9007199254740992.0)

    # ------------------------------------------- numpy-compatible surface

    def random(self, size: Optional[int] = None):
        if size is None:
            return self._next_double()
        return self._next_double_bulk(int(size))

    def _standard_normals(self, count: int) -> np.ndarray:
        """Box-Muller in pairs.

        The pairing is part of the contract: an odd count still consumes a whole
        pair's worth of uniforms and discards the second value, so the next call
        starts at the same point in the stream in both languages.
        """
        if count <= 0:
            return np.zeros(0)
        pairs = (count + 1) // 2
        uniforms = self._next_double_bulk(pairs * 2)
        u1 = np.maximum(uniforms[0::2], 1e-300)
        u2 = uniforms[1::2]
        radius = np.sqrt(-2.0 * np.log(u1))
        theta = 2.0 * np.pi * u2
        out = np.empty(pairs * 2)
        out[0::2] = radius * np.cos(theta)
        out[1::2] = radius * np.sin(theta)
        return out[:count]

    def normal(self, loc: float = 0.0, scale: float = 1.0, size: Optional[int] = None):
        if size is None:
            return float(loc + scale * self._standard_normals(1)[0])
        return loc + scale * self._standard_normals(int(size))

    def integers(self, high: int, size: Optional[int] = None):
        """Uniform integer in ``[0, high)`` by rejection, so the result is exact."""
        def draw() -> int:
            bound = int(high)
            threshold = (-bound) % bound
            while True:
                value = self._next_uint32()
                if value >= threshold:
                    return value % bound

        if size is None:
            return draw()
        return np.array([draw() for _ in range(int(size))])

    def choice(self, a: int, size: Optional[int] = None, p: Optional[Sequence[float]] = None):
        if p is None:
            return self.integers(a, size)
        weights = np.asarray(p, dtype=float)

        def draw() -> int:
            total = float(weights.sum())
            if total <= 0 or not math.isfinite(total):
                return int(self.integers(len(weights)))
            target = self._next_double() * total
            running = 0.0
            for index, weight in enumerate(weights):
                running += float(weight)
                if target < running:
                    return index
            return len(weights) - 1

        if size is None:
            return draw()
        return np.array([draw() for _ in range(int(size))])

    def multinomial(self, n: int, pvals: Sequence[float]) -> np.ndarray:
        """Counts of ``n`` independent categorical draws.

        Deliberately not numpy's conditional-binomial algorithm: drawing one
        category at a time is trivial to mirror exactly in Swift, and at the
        sizes used here - a few thousand particles, once per initialisation -
        the cost is irrelevant.
        """
        weights = np.asarray(pvals, dtype=float)
        counts = np.zeros(len(weights), dtype=np.int64)
        for _ in range(int(n)):
            counts[self.choice(len(weights), p=weights)] += 1
        return counts


def make_rng(seed: int, mode: str = "numpy"):
    """Build the generator the particle filter should use.

    ``numpy`` is the default and is what the baseline has always used.
    ``parity`` switches to the cross-language generator so a run can be compared
    against the Swift implementation draw for draw.
    """
    if mode == "parity":
        return ParityGenerator(seed)
    if mode == "numpy":
        return np.random.default_rng(seed)
    raise ValueError(f"unknown rng mode {mode!r}; expected 'numpy' or 'parity'")
