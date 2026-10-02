"""Contrat HTTP des retours utilisateurs sur les chunks cités (lecture seule : nécessite pydantic)."""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class DocumentEvaluationCreateInput(BaseModel):
    """Contrat HTTP de création d'un retour utilisateur sur la pertinence d'un chunk."""

    source: str = Field(min_length=1, max_length=255)
    document_id: int
    chunk_index: Optional[int] = None
    chat_id: str = Field(min_length=1, max_length=256)
    message_id: str = Field(min_length=1, max_length=256)
    eval: str = Field(min_length=1, max_length=64)
    comment: Optional[str] = Field(default=None, max_length=4096)


class DocumentEvaluationResponse(BaseModel):
    id: int
    source_id: int
    document_id: int
    chunk_index: Optional[int] = None
    chat_id: str
    message_id: str
    eval: str
    comment: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


# Table SQL correspondante (extrait du modèle ORM) : document_evaluation
#   id BIGINT PK | source_id INT FK source.id | document_id BIGINT FK document.id
#   chunk_index INT NULL | chat_id VARCHAR(256) | message_id VARCHAR(256)
#   eval VARCHAR(64) | comment VARCHAR(4096) NULL | created_at, updated_at TIMESTAMP
#
# Ces retours sont saisis par les utilisateurs dans l'interface de chat (pouce haut/bas
# sur un document cité). Exemple d'extrait : data/document_evaluations.jsonl
