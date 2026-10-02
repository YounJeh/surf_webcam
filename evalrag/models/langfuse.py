from dataclasses import dataclass, asdict
from typing import Any, Literal

_TypeInteraction = Literal["generation", "embedding", "span", "agent", "tool", "chain", "retriever", "evaluator", "guardrail"]

@dataclass
class LangfuseInput:
    prompt: str | None = None
    histo_msg: list[dict] | None = None
    histo_size: int | None = None
    query: str | None = None
    num_texts: int | None = None
    model: str | None = None
    nb_variants: int | None = None
    num_rag_chunks: int | None = None
    chat_id: str | None = None
    message_id: str | None = None
    user: dict | None = None

@dataclass
class LangfuseObservation:
    trace_context: dict[str, Any] | None = None
    as_type: _TypeInteraction | None = None
    name: str | None = None
    input: LangfuseInput | None = None
    metadata: dict[str, Any] | None = None
    prompt: str | None = None
    response: str | None = None
    contexts: list[str] | None = None
    expected_response: str | None = None
    source: str | None = None  # dataset | openwebui | api
    conversation_id: str | None = None  # chat_id, run_id, session_id

    def to_langfuse(self) -> dict[str, Any]:
        """Retourne un dict propre, sans None"""
        raw = asdict(self)
        cleaned = _drop_none(raw)
        return cleaned or {}

def _drop_none(obj: list | dict) -> list | dict:
    """Méthode interne, supprime récursivement les None dans dict/list."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            v2 = _drop_none(v)
            if v2 is not None:
                out[k] = v2
        return out or None
    if isinstance(obj, list):
        out = []
        for v in obj:
            v2 = _drop_none(v)
            if v2 is not None:
                out.append(v2)
        return out or None
    return obj
