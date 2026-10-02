"""Modèles HTTP de l'API d'évaluation (lecture seule : nécessite pydantic, non importé par les runners)."""
from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel, Field


class EvalRunRequest(BaseModel):
    run_name: str = Field(default="rag-demo-eval")
    mode: str = Field(default="fast", pattern="^(fast|full)$")
    rows_concurrency: int = Field(default=4)
    metrics_concurrency: int = Field(default=2)
    config_path: str = Field(default="config/eval.local.ini")
    config_overrides: dict[str, dict[str, str | int | float | bool]] = Field(default_factory=dict)


class EvalRunResponse(BaseModel):
    run_id: str
    status: str
    run_name: str
    log_path: str | None = None
    created_at: datetime

class EvalRunStatusResponse(BaseModel):
    run_id: str
    status: str
    run_name: str
    log_path: str | None = None
    created_at: datetime
    return_code: int | None = None