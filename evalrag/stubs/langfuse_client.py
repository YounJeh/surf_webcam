"""
[STUB] Client Langfuse en mémoire, persistant sur disque (dossier out/langfuse/).

Remplace le SDK Langfuse (serveur d'observabilité). Seul le sous-ensemble d'API utilisé
par le pipeline d'évaluation est reproduit :
- datasets : get_dataset, create_dataset, create_dataset_item ;
- runs : DatasetItem.run(...) -> span avec score_trace / update ;
- lecture : get_run, get_trace (utilisé par le rapport de comparaison) ;
- traçage : start_as_current_observation (utilisé par EvaluationTracer).

Il n'est pas nécessaire de lire ce fichier pour répondre aux questions.
"""
from __future__ import annotations

import json
import re
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evalrag.config import PACKAGE_ROOT

OUT_DIR = PACKAGE_ROOT / "out" / "langfuse"
DATA_DIR = PACKAGE_ROOT / "data"
SEED_FILE = DATA_DIR / "golden_dataset.jsonl"
DEMO_DATASET = "golden_set_demo"


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


class _Span:
    def __init__(self, trace: dict[str, Any]):
        self._trace = trace

    def update(self, **kwargs):
        self._trace.setdefault("updates", []).append(kwargs)
        for key in ("input", "output", "status"):
            if key in kwargs:
                self._trace[key] = kwargs[key]

    def update_trace(self, **kwargs):
        self._trace.setdefault("trace_attributes", {}).update(kwargs)

    def score_trace(self, name: str, value: float, data_type: str = "NUMERIC"):
        self._trace["scores"].append({"name": name, "value": value, "data_type": data_type})


@dataclass
class StubDatasetItem:
    id: str
    input: Any
    expected_output: Any
    metadata: dict | None
    _client: "StubLangfuse" = field(repr=False)

    @contextmanager
    def run(self, run_name: str, run_description: str | None = None, run_metadata: dict | None = None):
        trace = self._client._new_trace(name=f"dataset-run:{run_name}", metadata=run_metadata)
        self._client._link_run_item(run_name, run_description, run_metadata, self.id, trace["id"])
        yield _Span(trace)


@dataclass
class StubDataset:
    name: str
    items: list[StubDatasetItem]


class StubLangfuse:
    def __init__(self):
        self._datasets: dict[str, list[dict]] = {}
        self._traces: dict[str, dict] = {}
        self._runs: dict[str, dict] = {}

    # ---- datasets ---------------------------------------------------------
    def create_dataset(self, name: str, description: str | None = None, metadata: dict | None = None):
        self._datasets.setdefault(name, [])

    def create_dataset_item(self, dataset_name: str, input: Any, expected_output: Any = None, metadata: dict | None = None):
        items = self._datasets.setdefault(dataset_name, [])
        items.append({
            "id": f"item-{len(items) + 1:03d}",
            "input": input,
            "expected_output": expected_output,
            "metadata": metadata,
        })

    def get_dataset(self, name: str) -> StubDataset:
        path = OUT_DIR / "datasets" / f"{_safe(name)}.json"
        if name not in self._datasets and path.exists():
            self._datasets[name] = json.loads(path.read_text(encoding="utf-8"))
        if name not in self._datasets and name != DEMO_DATASET:
            raise KeyError(f"Dataset introuvable : {name}")
        if name not in self._datasets:
            # Premier lancement : le dataset est importé depuis le fichier métier,
            # avec le même code que l'import réel (evalrag.dataset_import).
            from evalrag.dataset_import import load_rows, import_rows
            self.create_dataset(name)
            import_rows(self, name, load_rows(SEED_FILE))
            self._persist_dataset(name)
        raw = self._datasets[name]
        return StubDataset(
            name=name,
            items=[StubDatasetItem(r["id"], r["input"], r["expected_output"], r["metadata"], self) for r in raw],
        )

    def _persist_dataset(self, name: str):
        path = OUT_DIR / "datasets" / f"{_safe(name)}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._datasets[name], ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- traces / runs ----------------------------------------------------
    def _new_trace(self, name: str, metadata: dict | None = None) -> dict:
        trace_id = uuid.uuid4().hex
        trace = {"id": trace_id, "name": name, "metadata": metadata, "scores": [], "observations": []}
        self._traces[trace_id] = trace
        return trace

    def _link_run_item(self, run_name, run_description, run_metadata, dataset_item_id, trace_id):
        run = self._runs.setdefault(run_name, {
            "name": run_name, "description": run_description, "metadata": run_metadata, "dataset_run_items": [],
        })
        run["dataset_run_items"].append({"dataset_item_id": dataset_item_id, "trace_id": trace_id})

    @contextmanager
    def start_as_current_observation(self, **kwargs):
        trace = self._new_trace(name=kwargs.get("name", "observation"), metadata=kwargs.get("metadata"))
        trace["observation"] = {k: v for k, v in kwargs.items() if k != "metadata"}
        yield _Span(trace)

    def create_score(self, name: str, value: float, trace_id: str, **kwargs):
        self._traces[trace_id]["scores"].append({"name": name, "value": value})

    def get_run(self, dataset_name: str, run_name: str) -> dict:
        path = OUT_DIR / "runs" / f"{_safe(run_name)}.json"
        if not path.exists():
            raise FileNotFoundError(f"Run introuvable : {run_name} ({path})")
        return json.loads(path.read_text(encoding="utf-8"))

    def get_trace(self, trace_id: str) -> dict:
        path = OUT_DIR / "traces" / f"{trace_id}.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def flush(self):
        for name in self._datasets:
            self._persist_dataset(name)
        (OUT_DIR / "runs").mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "traces").mkdir(parents=True, exist_ok=True)
        run_trace_ids = set()
        for run in self._runs.values():
            path = OUT_DIR / "runs" / f"{_safe(run['name'])}.json"
            path.write_text(json.dumps(run, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            run_trace_ids.update(i["trace_id"] for i in run["dataset_run_items"])
            print(f"[stub-langfuse] run enregistré : {run['name']} -> {path}", flush=True)
        for trace_id in run_trace_ids:
            trace = self._traces[trace_id]
            path = OUT_DIR / "traces" / f"{trace_id}.json"
            path.write_text(json.dumps(trace, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


_CLIENT: StubLangfuse | None = None


def get_langfuse_client() -> StubLangfuse:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = StubLangfuse()
    return _CLIENT
