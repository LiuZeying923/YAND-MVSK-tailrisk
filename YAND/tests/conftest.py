"""Shared fixtures.  Everything runs offline on deterministic synthetic data."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(20240101)


@pytest.fixture
def returns(rng):
    """A (T, n) synthetic return matrix with mild fat tails."""
    T, n = 300, 10
    return rng.standard_t(df=6, size=(T, n)) * 0.015 + 0.0005


@pytest.fixture
def prices(rng):
    """A (T, n) positive price matrix, ~2 years daily, 8 assets."""
    steps = rng.standard_t(df=6, size=(520, 8)) * 0.014 + 0.0004
    return 100.0 * np.cumprod(1.0 + steps, axis=0)
