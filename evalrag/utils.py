from datetime import datetime
import json

from evalrag import config as config_utils


def log_step(message: str, emoji: str = ""):
    ts = datetime.now().strftime("%H:%M:%S")
    prefix = f"{emoji} " if emoji else ""
    try:
        print(f"{prefix}[{ts}] {message}", flush=True)
    except UnicodeEncodeError:
        print(f"[{ts}] {message}", flush=True)


def extract_text_from_chunk(chunk) -> str:
    if chunk is None:
        return ""

    if isinstance(chunk, dict):
        for k in ("text", "content", "chunk", "body"):
            if k in chunk and isinstance(chunk[k], str) and chunk[k].strip():
                return chunk[k]
        return json.dumps(chunk, ensure_ascii=False)

    for attr in ("text", "content", "chunk", "body"):
        if hasattr(chunk, attr):
            val = getattr(chunk, attr)
            if isinstance(val, str) and val.strip():
                return val

    return str(chunk)


def get_rag_run_config() -> dict:
    return {
        "database": {
            "pg_database": config_utils.get_str("Database", "PG_DATABASE", ""),
            "vector_size": config_utils.get_int("Database", "VECTOR_SIZE", 0),
        },
        "embedding": {
            "server": config_utils.get_str("Embedding", "SERVER", ""),
            "default_model": config_utils.get_str("Embedding", "DEFAULT_MODEL", ""),
            "batch_size": config_utils.get_int("Embedding", "BATCH_SIZE", 0),
            "chunk_size": config_utils.get_int("Embedding", "CHUNK_SIZE", 0),
            "chunk_overlap": config_utils.get_int("Embedding", "CHUNK_OVERLAP", 0),
            "top_k": config_utils.get_int("Embedding", "TOP_K", 0),
            "timeout": config_utils.get_int("Embedding", "TIMEOUT", 0),
            "tolower": config_utils.get_bool("Embedding", "TOLOWER", False),
            "purge_delay": config_utils.get_int("Embedding", "PURGE_DELAY", 0),
        },
        "glossaire": {
            "glossaire_path": config_utils.get_int("Glossaire", "FUSIONNE_PATH", 0),
            "glossaire_path": config_utils.get_int("Glossaire", "CRISTAL_PATH", 0),
        },
        "embedding_source": {
            "default_model": config_utils.get_str("Embedding_source", "DEFAULT_MODEL", ""),
            "chunk_size": config_utils.get_int("Embedding_source", "CHUNK_SIZE", 0),
            "chunk_overlap": config_utils.get_int("Embedding_source", "CHUNK_OVERLAP", 0),
            "top_k": config_utils.get_int("Embedding_source", "TOP_K", 0),
        },
        "retriever": {
            "nb_db_results": config_utils.get_int("Retriever", "NB_DB_RESULTS", 0),
            "rrf_min_results": config_utils.get_int("Retriever", "RRF_MIN_RESULTS", 0),
            "rrf_max_results": config_utils.get_int("Retriever", "RRF_MAX_RESULTS", 0),
            "use_reranking": config_utils.get_bool("Retriever", "USE_RERANKING", False),
            "mmr_lambda": config_utils.get_float("Retriever", "MMR_LAMBDA", 0.0),
            "nb_results": config_utils.get_int("Retriever", "NB_RESULTS", 0),
            "max_semantic_distance": config_utils.get_int("Retriever", "MAX_SEMANTIC_DISTANCE", 0),
            "neighbor_chunks": config_utils.get_int("Retriever", "NEIGHBOR_CHUNKS", 0),
            "use_hybrid_retrieval": config_utils.get_bool("Retriever", "USE_HYBRID_RETRIEVAL", False),
            "lexical_db_results": config_utils.get_int("Retriever", "LEXICAL_DB_RESULTS", 0),
            "lexical_trigram_threshold": config_utils.get_float("Retriever", "LEXICAL_TRIGRAM_THRESHOLD", 0.0),
            "lexical_fts_weight": config_utils.get_float("Retriever", "LEXICAL_FTS_WEIGHT", 0.0),
            "lexical_trigram_weight": config_utils.get_float("Retriever", "LEXICAL_TRIGRAM_WEIGHT", 0.0),
        },
        "multiquery": {
            "server": config_utils.get_str("Multiquery", "SERVER", ""),
            "default_model": config_utils.get_str("Multiquery", "DEFAULT_MODEL", ""),
            "timeout": config_utils.get_int("Multiquery", "TIMEOUT", 0),
            "temperature": config_utils.get_float("Multiquery", "TEMPERATURE", 0.0),
            "top_p": config_utils.get_float("Multiquery", "TOP_P", 1.0),
            "nb_variants": config_utils.get_int("Multiquery", "NB_VARIANTS", 0),
        },
        "reranking": {
            "server": config_utils.get_str("Reranking", "SERVER", ""),
            "default_model": config_utils.get_str("Reranking", "DEFAULT_MODEL", ""),
            "batch_size": config_utils.get_int("Reranking", "BATCH_SIZE", 0),
            "timeout": config_utils.get_int("Reranking", "TIMEOUT", 0),
            "score_minimum": config_utils.get_float("Reranking", "SCORE_MINIMUM", 0.0),
        },
        "response": {
            "server": config_utils.get_str("Response", "SERVER", ""),
            "default_model": config_utils.get_str("Response", "DEFAULT_MODEL", ""),
            "timeout": config_utils.get_int("Response", "TIMEOUT", 0),
            "temperature": config_utils.get_float("Response", "TEMPERATURE", 0.0),
            "history_size": config_utils.get_int("Response", "HISTORY_SIZE", 0),
        },
    }


def build_run_description(source_name: str, rag_cfg: dict) -> str:
    return (
        f"Évaluation RAG {source_name} | "
        f"embed_model={rag_cfg['embedding_source']['default_model']} | "
        f"chunk_size/overlap={rag_cfg['embedding_source']['chunk_size']}/{rag_cfg['embedding_source']['chunk_overlap']} | "
        f"top_k={rag_cfg['embedding_source']['top_k']} | "
        f"nb_db_results={rag_cfg['retriever']['nb_db_results']} | "
        f"hybrid={rag_cfg['retriever']['use_hybrid_retrieval']} | "
        f"neighbor_chunks={rag_cfg['retriever']['neighbor_chunks']} | "
        f"rerank={rag_cfg['retriever']['use_reranking']} | "
        f"reranker_model={rag_cfg['reranking']['default_model']} | "
        f"response_model={rag_cfg['response']['default_model']} | "
        f"response_temperature={rag_cfg['response']['temperature']} | "
        f"multiquery_nb_variants={rag_cfg['multiquery']['nb_variants']}"
    )
