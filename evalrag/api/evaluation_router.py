"""Routes FastAPI de lancement d'évaluation (lecture seule : nécessite fastapi, non importé par les runners)."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from evalrag.api.api_models import (
    EvalRunRequest,
    EvalRunResponse,
    EvalRunStatusResponse,
)
from evalrag.services.eval_launcher import EvalLauncher

router = APIRouter(prefix="/evaluation", tags=["Evaluation"])

launcher = EvalLauncher()


@router.post("/runs", response_model=EvalRunResponse)
async def launch_evaluation_run(payload: EvalRunRequest) -> EvalRunResponse:
    """
    Lance un run d'évaluation en arrière-plan via subprocess.
    La requête HTTP retourne immédiatement.
    """
    try:
        result = launcher.launch_run(
            run_name=payload.run_name,
            mode=payload.mode,
            rows_concurrency=payload.rows_concurrency,
            metrics_concurrency=payload.metrics_concurrency,
            config_path=payload.config_path,
            config_overrides=payload.config_overrides,
        )

        return EvalRunResponse(**result)

    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du lancement du run : {exc}"
        ) from exc

@router.get("/runs/{run_id}", response_model=EvalRunStatusResponse)
async def get_evaluation_run_status(run_id: str) -> EvalRunStatusResponse:
    """
    Retourne le statut d'un run déjà lancé.
    """
    result = launcher.get_run_status(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Run introuvable : {run_id}")

    return EvalRunStatusResponse(**result)