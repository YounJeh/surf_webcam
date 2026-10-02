"""
[STUB] Services RAG : retriever et générateur de réponse.

En production :
- RetrieverService interroge la base vectorielle (pgvector) + recherche lexicale,
  fusionne (RRF), reranke, puis filtre par distance sémantique ;
- ResponseGeneratorService appelle le LLM de génération avec les chunks en contexte.

Ici :
- le retriever relit des résultats pré-calculés (data/retrieval_fixture.json : chunks
  candidats et distance cosinus), applique le filtre [Retriever] MAX_SEMANTIC_DISTANCE
  puis coupe à [Retriever] NB_RESULTS ;
- le générateur relit des réponses pré-enregistrées (data/generation_fixture.json).
  Une variante peut exiger la présence de certains chunks dans le contexte. S'il reste
  plusieurs variantes possibles et que [Response] TEMPERATURE > 0, l'une est tirée au
  hasard (simulation d'un LLM échantillonné).

Il n'est pas nécessaire de lire ce fichier pour répondre aux questions.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from typing import List, Optional

from evalrag import config as config_utils
from evalrag.config import PACKAGE_ROOT
from evalrag.models.langfuse import LangfuseInput, LangfuseObservation
from evalrag.tracing.tracer import EvaluationTracer

DATA_DIR = PACKAGE_ROOT / "data"


@dataclass
class QueryResult:
    chunk_id: str
    document_id: int
    document_path: str
    chunk_index: int
    text: str
    score: float  # 1 - distance cosinus


def _load_jsonl(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _question_index() -> dict[str, str]:
    rows = _load_jsonl(DATA_DIR / "golden_dataset.jsonl")
    return {r["request"].strip(): r["request_id"] for r in rows}


class RetrieverService:
    def __init__(self, source_name: str) -> None:
        self.sourceName = source_name
        self.corpus = {c["chunk_id"]: c for c in _load_jsonl(DATA_DIR / "corpus.jsonl")}
        self.fixture = json.loads((DATA_DIR / "retrieval_fixture.json").read_text(encoding="utf-8"))
        self.questions = _question_index()

    async def retrieve_chunks(self, prompt: str, histo_msg: Optional[List[dict]] = None) -> List[QueryResult]:
        if not prompt:
            return []

        tracer = EvaluationTracer()
        span_obs = LangfuseObservation(as_type="retriever", name="retrieval", input=LangfuseInput(prompt=prompt))

        with tracer.observe(span_obs) as span:
            start_time = time.perf_counter()
            max_distance = config_utils.get_float("Retriever", "MAX_SEMANTIC_DISTANCE", 1.0)
            nb_results = config_utils.get_int("Retriever", "NB_RESULTS", 10)

            request_id = self.questions.get(prompt.strip())
            candidates = self.fixture.get(request_id, []) if request_id else []
            candidates = sorted(candidates, key=lambda c: c["distance"])
            kept = [c for c in candidates if c["distance"] <= max_distance][:nb_results]

            out = []
            for c in kept:
                chunk = self.corpus[c["chunk_id"]]
                out.append(QueryResult(
                    chunk_id=chunk["chunk_id"],
                    document_id=chunk["document_id"],
                    document_path=chunk["document_path"],
                    chunk_index=chunk["chunk_index"],
                    text=chunk["text"],
                    score=round(1 - c["distance"], 4),
                ))

            span.update(output={
                "num_chunks": len(out),
                "retrieval_time_sec": time.perf_counter() - start_time,
                "chunks": [{"chunk_id": q.chunk_id, "score": q.score} for q in out],
            })

        return out


class ResponseGeneratorService:
    def __init__(self, source_name: str) -> None:
        self.sourceName = source_name
        self.fixture = json.loads((DATA_DIR / "generation_fixture.json").read_text(encoding="utf-8"))
        self.questions = _question_index()

    async def generate_response(self, question: str, rag_results: List[QueryResult],
                                histo_msg: Optional[List[dict]] = None) -> str:
        temperature = config_utils.get_float("Response", "TEMPERATURE", 0.0)
        present = {r.chunk_id for r in rag_results}
        request_id = self.questions.get(question.strip())
        variants = self.fixture.get(request_id, []) if request_id else []

        eligible = [v for v in variants if all(c in present for c in v.get("requires", []))]
        if not eligible:
            return "Je ne sais pas."

        most_specific = max(len(v.get("requires", [])) for v in eligible)
        eligible = [v for v in eligible if len(v.get("requires", [])) == most_specific]

        chosen = random.choice(eligible) if temperature > 0 else eligible[0]
        return chosen["text"]
