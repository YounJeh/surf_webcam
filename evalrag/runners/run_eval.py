import argparse
import asyncio
import configparser
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List

from evalrag.config import DEFAULT_CONFIG, load_config
from evalrag.utils import extract_text_from_chunk, log_step, build_run_description, get_rag_run_config
from evalrag.metrics.ragas_metrics import build_metrics, build_ragas_clients, score_with_ragas

# [STUB] En production : client Langfuse, RetrieverService (pgvector + lexical + rerank)
# et ResponseGeneratorService (LLM via endpoint compatible OpenAI).
from evalrag.stubs.langfuse_client import get_langfuse_client
from evalrag.stubs.rag_services import ResponseGeneratorService, RetrieverService

# CONSTANTES
SOURCE_NAME = "base_documentaire"
LANGFUSE_DATASET_NAME = "golden_set_demo"

# PROCESS

async def _process_item(
    *,
    item,
    run_name: str,
    row_index: int,
    retriever: RetrieverService,
    responder: ResponseGeneratorService,
    metrics,
    row_sem: asyncio.Semaphore,
    metrics_concurrency,
) -> Dict[str, Any]:
    """
    Traite un DatasetItem Langfuse :
      - lie automatiquement la trace à l'item via item.run()
      - attache les scores RAGAS directement sur le run
    """
    prompt = item.input
    expected_response = str(item.expected_output)
    rag_cfg = get_rag_run_config()
    run_description = build_run_description(SOURCE_NAME, rag_cfg)


    async with row_sem:
        row_t0 = time.perf_counter()
        log_step(f"[row {row_index:03d}] START", "🚀")

        with item.run(
            run_name=run_name,
            run_description=run_description,
            run_metadata={"source": SOURCE_NAME,
                          "rag_config": rag_cfg},
        ) as item_span:

            try:
                t_retrieve = time.perf_counter()
                chunks = await retriever.retrieve_chunks(prompt=prompt) or []
                ctx_texts = [extract_text_from_chunk(c) for c in chunks]
                retrieve_s = time.perf_counter() - t_retrieve

                t_generate = time.perf_counter()
                response = await responder.generate_response(
                    question=prompt,
                    rag_results=chunks,
                    histo_msg=None,
                )
                generate_s = time.perf_counter() - t_generate

                t_score = time.perf_counter()
                scores = await score_with_ragas(metrics, prompt, ctx_texts, expected_response, response, metrics_concurrency)
                score_s = time.perf_counter() - t_score

                for metric_name, value in scores.items():
                    if value is not None:
                        item_span.score_trace(
                            name=metric_name,
                            value=value,
                            data_type="NUMERIC",
                        )

                item_span.update(
                    input=prompt,
                    output=response,
                )
                item_span.score_trace(
                    name="nb_error",
                    value=0,
                    data_type="NUMERIC",
                )

                total_s = time.perf_counter() - row_t0
                log_step(
                    f"[row {row_index:03d}] OK | ctx={len(ctx_texts)} | "
                    f"retrieve={retrieve_s:.1f}s | generate={generate_s:.1f}s | "
                    f"score={score_s:.1f}s | total={total_s:.1f}s",
                    "✅",
                )

                return {
                    "ok": True,
                    "row_index": row_index,
                    "ctx_len": len(ctx_texts),
                    "scores": scores,
                }

            except Exception as e:
                tb = traceback.format_exc()

                item_span.update(
                    status="ERROR",
                    output={"error": str(e), "traceback": tb},
                )
                item_span.score_trace(
                    name="nb_error",
                    value=1,
                    data_type="NUMERIC",
                )

                total_s = time.perf_counter() - row_t0
                log_step(
                    f"[row {row_index:03d}] ERROR après {total_s:.1f}s : {e}",
                    "❌",
                )
                print(tb, flush=True)

                return {
                    "ok": False,
                    "row_index": row_index,
                    "error": str(e),
                }


async def run_rag_with_ragas(
    mode: str = "full",
    rows_concurrency: int = 4,
    metrics_concurrency: int = 2,
    run_name: str = "rag-demo-eval",
    config_path: str = str(DEFAULT_CONFIG),
):
    load_config(config_path)
    langfuse = get_langfuse_client()

    dataset = langfuse.get_dataset(LANGFUSE_DATASET_NAME)
    items = dataset.items   #[:10] pour faire des tests
    total = len(items)
    log_step(f"Dataset '{LANGFUSE_DATASET_NAME}' chargé ({total} lignes)", "📦")

    unique_run_name = f"{run_name}-{datetime.now().isoformat()}"

    retriever = RetrieverService(source_name=SOURCE_NAME)
    responder = ResponseGeneratorService(source_name=SOURCE_NAME)
    log_step("Services Retriever + ResponseGenerator initialisés", "🔧")

    cfg = configparser.ConfigParser()
    cfg.read(config_path)
    llm, emb = build_ragas_clients(cfg)
    metrics = build_metrics(llm, emb, mode=mode)

    t0 = time.time()
    log_step(
        f"Début du run '{run_name}' | rows={total} | mode={mode} | "
        f"rows_concurrency={rows_concurrency} | metrics_concurrency={metrics_concurrency}",
        "🏁",
    )

    row_sem = asyncio.Semaphore(rows_concurrency)
    results: List[Dict[str, Any]] = []

    def launch(row_index: int, item) -> asyncio.Task:
        return asyncio.create_task(
            _process_item(
                item=item,
                run_name=unique_run_name,
                row_index=row_index,
                retriever=retriever,
                responder=responder,
                metrics=metrics,
                row_sem=row_sem,
                metrics_concurrency=metrics_concurrency,
            )
        )

    enum_items = list(enumerate(items, start=1))

    completed = 0
    success_so_far = 0
    errors_so_far = 0

    for batch_start in range(0, len(enum_items), rows_concurrency):
        batch = enum_items[batch_start: batch_start + rows_concurrency]

        log_step(
            f"Lancement batch rows {batch[0][0]} à {batch[-1][0]} "
            f"({len(batch)} tâches)",
            "🚦",
        )

        tasks = [launch(row_index, item) for row_index, item in batch]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in batch_results:
            if isinstance(r, Exception):
                tb = "".join(traceback.format_exception(type(r), r, r.__traceback__))
                log_step(f"Erreur non interceptée dans une task : {r}", "💥")
                print(tb, flush=True)
                result = {
                    "ok": False,
                    "row_index": None,
                    "error": str(r),
                }
            else:
                result = r

            results.append(result)
            completed += 1

            if result.get("ok"):
                success_so_far += 1
            else:
                errors_so_far += 1

        log_step(
            f"Fin batch | progression {completed}/{total} | "
            f"success={success_so_far} | errors={errors_so_far}",
            "📌",
        )

    log_step(
        f"Run terminé en {time.time() - t0:.1f}s",
        "🏁",
    )

    langfuse.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG evaluation")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG), help="Chemin du fichier .ini")
    parser.add_argument(
        "--mode",
        choices=["full", "fast"],
        default="full",
        help="Mode d'évaluation : full=8 métriques, fast=4 métriques",
    )
    parser.add_argument(
        "--rows-concurrency",
        type=int,
        default=4,
        help="Nombre de rows traitées en parallèle",
    )
    parser.add_argument(
        "--metrics-concurrency",
        type=int,
        default=2,
        help="Nombre de métriques RAGAS évaluées en parallèle par row",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="rag-demo-eval",
        help="Nom du run Langfuse pour identifier l'évaluation",
    )

    args = parser.parse_args()

    try:
        asyncio.run(
            run_rag_with_ragas(
                mode=args.mode,
                rows_concurrency=args.rows_concurrency,
                metrics_concurrency=args.metrics_concurrency,
                run_name=args.run_name,
                config_path=args.config,
            )
        )
    except KeyboardInterrupt:
        log_step("Interruption utilisateur", "🛑")
    except Exception as e:
        log_step(f"Erreur fatale : {e}", "💥")
        raise
