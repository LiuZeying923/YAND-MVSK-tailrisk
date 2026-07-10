"""Pydantic request/response models for the API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class GammaRequest(BaseModel):
    answers: dict[str, str] = Field(..., description="VBQ-5 answers, e.g. {'Q1':'B',...}")
    base_gamma: float = Field(6.0, ge=1.5, le=20.0)


class OptimizeRequest(BaseModel):
    tickers: list[str] = Field(..., min_length=1, max_length=60)
    gamma: float | None = Field(None, ge=1.0, le=25.0)
    answers: dict[str, str] | None = None
    base_gamma: float = Field(6.0, ge=1.5, le=20.0)
    max_weight: float = Field(0.35, gt=0.0, le=1.0)
    horizon: int = Field(21, ge=1, le=63, description="Optimisation return horizon in trading days")
    lookback: str = Field("3y", description="History window for the live fetch")
    offline: bool = Field(False, description="Force deterministic synthetic data")
    apply_guard: bool = Field(True, description="Re-optimise with the guard's gamma multiplier if tail risk is elevated")
    rf: float = Field(0.02, ge=0.0, le=0.2, description="Annual risk-free rate for Sharpe/Sortino")


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
