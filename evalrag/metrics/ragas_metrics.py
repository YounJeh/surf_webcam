import asyncio
import configparser
import inspect
import traceback
from dataclasses import dataclass
from typing import List, Optional

from evalrag import config as config_utils
from evalrag.utils import log_step
from evalrag.metrics.constants import RAGAS_METRIC_FACTORIES, METRICS_BY_MODE

# [STUB] En production : clients AsyncOpenAI (httpx, certificat PEM, proxy) vers le juge
# LLM et le serveur d'embeddings, enveloppés par ragas.llms.llm_factory et
# ragas.embeddings.OpenAIEmbeddings. Ici : équivalents locaux sans réseau.
from evalrag.stubs.llm_judge import StubJudgeLLM
from evalrag.stubs.embeddings import StubEmbeddings


@dataclass
class SingleTurnSample:
    """Équivalent de ragas.dataset_schema.SingleTurnSample."""

    user_input: str
    retrieved_contexts: List[str]
    response: str
    reference: Optional[str] = None


# INSTANCIATION RAGAS CLIENT

def build_ragas_clients(cfg: configparser.ConfigParser):
    log_step("Initialisation des clients RAGAS", "🛠️")
    judge_server = config_utils.get_str("Response", "SERVER", "")
    judge_model = config_utils.get_str("Response", "DEFAULT_MODEL", "")
    judge_base_url = judge_server.rstrip("/") + "/v1"

    emb_server = config_utils.get_str("Embedding", "SERVER", "")
    emb_model = config_utils.get_str("Embedding_source", "DEFAULT_MODEL", "")
    emb_base_url = emb_server.rstrip("/") + "/v1"

    # RAGAS LLM + Embeddings
    ragas_llm = StubJudgeLLM(model=judge_model, base_url=judge_base_url, temperature=0, max_tokens=4096)
    ragas_emb = StubEmbeddings(model=emb_model, base_url=emb_base_url)

    log_step("Clients RAGAS LLM + embeddings initialisés", "✅")
    return ragas_llm, ragas_emb


# INSTANCIATION METRIQUES

def build_metrics(ragas_llm, ragas_emb, mode: str = "full"):

    metric_names = METRICS_BY_MODE.get(mode)
    if metric_names is None:
        raise ValueError(f"Mode inconnu : '{mode}'. Modes disponibles : {list(METRICS_BY_MODE)}")

    metrics = []
    for name in metric_names:
        cls = RAGAS_METRIC_FACTORIES[name]
        # Certaines classes acceptent llm= en constructeur, d'autres non
        sig = inspect.signature(cls.__init__)
        kwargs = {}
        if "llm" in sig.parameters:
            kwargs["llm"] = ragas_llm
        if "embeddings" in sig.parameters:
            kwargs["embeddings"] = ragas_emb
        metrics.append(cls(**kwargs))

    log_step(f"Métriques RAGAS initialisées (mode={mode}, n={len(metrics)})", "📏")
    return metrics


# SCORING

async def call_ascore(m, sample):
    sig = inspect.signature(m.ascore)
    candidates = {
        "user_input": sample.user_input,
        "retrieved_contexts": sample.retrieved_contexts,
        "response": sample.response,
        "reference": sample.reference,
        # alias possibles selon versions
        "question": sample.user_input,
        "contexts": sample.retrieved_contexts,
        "answer": sample.response,
        "ground_truth": sample.reference,
    }

    kwargs = {k: v for k, v in candidates.items() if k in sig.parameters}
    return await m.ascore(**kwargs)


async def score_with_ragas(metrics, query, chunks, response, reference, max_concurrency=2):
    sample = SingleTurnSample(
        user_input=query,
        retrieved_contexts=chunks,
        response=response,
        reference=reference,
    )

    sem = asyncio.Semaphore(max_concurrency)

    async def run_metric(m):
        async with sem:
            return await call_ascore(m, sample)

    results = await asyncio.gather(*(run_metric(m) for m in metrics), return_exceptions=True)

    scores = {}
    for m, r in zip(metrics, results):
        if isinstance(r, Exception):
            log_step(
                f"Erreur métrique '{getattr(m, 'name', type(m).__name__)}' "
                f"({type(r).__name__}) : {r}",
                "⚠️"
            )
            print("".join(traceback.format_exception(type(r), r, r.__traceback__)), flush=True)
            scores[getattr(m, "name", type(m).__name__)] = None
        else:
            scores[getattr(m, "name", type(m).__name__)] = float(r) if r is not None else None
    return scores
