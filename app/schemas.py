from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class SourceSyncResponse(BaseModel):
    synced_articles: int
    cluster_count: int
    week_key: str


class IssueCreateRequest(BaseModel):
    week_key: str
    title: str
    topic_ids: list[str] = Field(default_factory=list)
    generate_now: bool = True


class GenerationRequest(BaseModel):
    regenerate: bool = False


class FeedbackSectionPayload(BaseModel):
    section_id: Optional[str] = None
    original_text: str
    edited_text: str
    chosen_text: Optional[str] = None
    rejected_text: Optional[str] = None
    reason: Optional[str] = None
    factuality: int = 3
    style: int = 3
    structure: int = 3
    publishability: int = 3
    flags: list[str] = Field(default_factory=list)
    notes: Optional[str] = None


class FeedbackRequest(BaseModel):
    verdict: str
    sections: list[FeedbackSectionPayload] = Field(default_factory=list)
    notes: Optional[str] = None


class CorpusImportResult(BaseModel):
    imported_documents: int
    style_profile_version: str


class TrainingRunResponse(BaseModel):
    artifact_id: str
    role: str
    status: str
    dataset_path: str
    metrics: dict[str, Any]


class EvalRunResponse(BaseModel):
    eval_run_id: str
    status: str
    summary: dict[str, Any]


class ModelSwitchRequest(BaseModel):
    artifact_id: str


class TopicResponse(BaseModel):
    id: str
    week_key: str
    title: str
    category: str
    score: float
    summary: str
    entities: list[str]
    article_ids: list[str]
    evidence_pack: dict[str, Any]
