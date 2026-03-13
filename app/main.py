from __future__ import annotations

from typing import Any, Optional
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import APP_TITLE, ROOT_DIR
from .db import init_db
from .schemas import (
    CorpusImportResult,
    EvalRunResponse,
    FeedbackRequest,
    GenerationRequest,
    IssueCreateRequest,
    ModelSwitchRequest,
    SourceSyncResponse,
    TopicResponse,
    TrainingRunResponse,
)
from .services.corpus import corpus_overview, import_uploads, latest_style_profile
from .services.editorial_ai import OpenAIEditorialClient
from .services.evaluation import run_evaluation
from .services.exports import build_download_filename, save_issue_download_copy
from .services.generation import create_issue, generate_issue, get_issue, list_issues
from .services.sources import get_topics, list_sources, seed_sources, sync_sources, topic_preference_overview, topics_overview
from .services.training import get_production_model, list_model_artifacts, rollback_model, run_preference_training, run_sft_training, set_production_model


app = FastAPI(title=APP_TITLE)
app.mount("/static", StaticFiles(directory=ROOT_DIR / "app" / "static"), name="static")
templates = Jinja2Templates(directory=str(ROOT_DIR / "app" / "templates"))


@app.on_event("startup")
def startup() -> None:
    init_db()
    seed_sources()


def render(request: Request, template: str, context: dict[str, Any]) -> HTMLResponse:
    base_context = {
        "request": request,
        "app_title": APP_TITLE,
        "current_path": request.url.path,
    }
    base_context.update(context)
    return templates.TemplateResponse(request, template, base_context)


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request) -> HTMLResponse:
    week_key = request.query_params.get("week")
    openai_status = OpenAIEditorialClient().status()
    return render(
        request,
        "dashboard.html",
        {
            "corpus": corpus_overview(),
            "topics": topics_overview(week_key),
            "topic_preferences": topic_preference_overview(),
            "issues": list_issues()[:8],
            "models": list_model_artifacts()[:6],
            "production_model": get_production_model(),
            "openai_status": openai_status,
        },
    )


@app.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request) -> HTMLResponse:
    openai_status = OpenAIEditorialClient().status()
    return render(
        request,
        "settings.html",
        {
            "profile": latest_style_profile(),
            "topic_preferences": topic_preference_overview(),
            "production_model": get_production_model(),
            "sources": list_sources(),
            "openai_status": openai_status,
        },
    )


@app.get("/corpus", response_class=HTMLResponse)
def corpus_page(request: Request) -> HTMLResponse:
    return render(request, "corpus.html", {"corpus": corpus_overview()})


@app.get("/sources", response_class=HTMLResponse)
def sources_page(request: Request) -> HTMLResponse:
    return render(
        request,
        "sources.html",
        {"sources": list_sources(), "topics": topics_overview(request.query_params.get("week"))},
    )


@app.get("/topics-board", response_class=HTMLResponse)
def topics_board(request: Request) -> HTMLResponse:
    week_key = request.query_params.get("week")
    return render(request, "topics.html", {"topics": topics_overview(week_key)})


@app.get("/issues/{issue_id}", response_class=HTMLResponse)
def issue_page(issue_id: str, request: Request) -> HTMLResponse:
    try:
        issue = get_issue(issue_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Issue not found") from exc
    return render(request, "issue.html", {"issue": issue})


@app.get("/training", response_class=HTMLResponse)
def training_page(request: Request) -> HTMLResponse:
    return render(
        request,
        "training.html",
        {
            "artifacts": list_model_artifacts(),
            "production_model": get_production_model(),
        },
    )


@app.get("/history", response_class=HTMLResponse)
def history_page(request: Request) -> HTMLResponse:
    return render(request, "history.html", {"issues": list_issues()})


@app.post("/corpus/import", response_model=CorpusImportResult)
async def import_corpus(files: list[UploadFile] = File(...)) -> CorpusImportResult:
    return CorpusImportResult(**(await import_uploads(files)))


@app.post("/sources/sync", response_model=SourceSyncResponse)
def sync_sources_endpoint() -> SourceSyncResponse:
    return SourceSyncResponse(**sync_sources())


@app.get("/topics", response_model=list[TopicResponse])
def topics_endpoint(week: Optional[str] = None) -> list[TopicResponse]:
    return [TopicResponse(**topic) for topic in get_topics(week)]


@app.post("/issues")
def create_issue_endpoint(payload: IssueCreateRequest) -> dict[str, Any]:
    issue = create_issue(payload.week_key, payload.title, payload.topic_ids)
    if payload.generate_now:
        return generate_issue(issue["id"], regenerate=True)
    return issue


@app.post("/issues/{issue_id}/generate")
def generate_issue_endpoint(issue_id: str, payload: Optional[GenerationRequest] = None) -> dict[str, Any]:
    payload = payload or GenerationRequest()
    try:
        return generate_issue(issue_id, payload.regenerate)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Issue not found") from exc


@app.post("/issues/{issue_id}/feedback")
def feedback_issue_endpoint(issue_id: str, payload: FeedbackRequest) -> dict[str, Any]:
    from .services.feedback import record_feedback

    try:
        return record_feedback(issue_id, payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Issue not found") from exc


@app.post("/training/sft", response_model=TrainingRunResponse)
def training_sft_endpoint() -> TrainingRunResponse:
    return TrainingRunResponse(**run_sft_training())


@app.post("/training/preferences", response_model=TrainingRunResponse)
def training_preferences_endpoint() -> TrainingRunResponse:
    return TrainingRunResponse(**run_preference_training())


@app.post("/eval/run", response_model=EvalRunResponse)
def evaluation_endpoint() -> EvalRunResponse:
    return EvalRunResponse(**run_evaluation())


@app.post("/models/promote")
def promote_model_endpoint(payload: ModelSwitchRequest) -> dict[str, Any]:
    try:
        return set_production_model(payload.artifact_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Artifact not found") from exc


@app.post("/models/rollback")
def rollback_model_endpoint(payload: ModelSwitchRequest) -> dict[str, Any]:
    try:
        return rollback_model(payload.artifact_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Artifact not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/issues/{issue_id}/export")
def export_issue_endpoint(issue_id: str, fmt: str = "md") -> Response:
    issue = get_issue(issue_id)
    if not issue["drafts"]:
        raise HTTPException(status_code=404, detail="Draft not found")
    latest = issue["drafts"][0]
    if fmt == "html":
        return Response(content=latest["html_content"], media_type="text/html; charset=utf-8")
    return Response(content=latest["markdown_content"], media_type="text/markdown; charset=utf-8")


@app.post("/ui/corpus/import")
async def corpus_form_upload(files: list[UploadFile] = File(...)) -> RedirectResponse:
    await import_uploads(files)
    return RedirectResponse(url="/corpus", status_code=303)


@app.post("/ui/sources/sync")
def sources_form_sync() -> RedirectResponse:
    sync_sources()
    return RedirectResponse(url="/topics-board", status_code=303)


@app.post("/ui/issues/create")
def issues_form_create(
    week_key: str = Form(...),
    title: str = Form(...),
    topic_ids: list[str] = Form(default_factory=list),
    generate_now: bool = Form(True),
) -> RedirectResponse:
    issue = create_issue(week_key, title, topic_ids)
    if generate_now:
        generate_issue(issue["id"], regenerate=True)
    return RedirectResponse(url=f"/issues/{issue['id']}", status_code=303)


@app.post("/ui/issues/create-download")
def issues_form_create_download(
    week_key: str = Form(...),
    title: str = Form(...),
    topic_ids: list[str] = Form(default_factory=list),
    generate_now: bool = Form(True),
    download_format: str = Form("md"),
) -> Response:
    issue = create_issue(week_key, title, topic_ids)
    issue_payload = (
        generate_issue(issue["id"], regenerate=True)
        if generate_now
        else get_issue(issue["id"])
    )
    if not issue_payload["drafts"]:
        return RedirectResponse(url=f"/issues/{issue['id']}", status_code=303)

    fmt = "html" if download_format == "html" else "md"
    latest = issue_payload["drafts"][0]
    body = latest["html_content"] if fmt == "html" else latest["markdown_content"]
    media_type = "text/html; charset=utf-8" if fmt == "html" else "text/markdown; charset=utf-8"
    filename = build_download_filename(issue_payload["week_key"], issue_payload["title"], fmt)
    save_issue_download_copy(filename, body)
    return Response(
        content=body,
        media_type=media_type,
        headers={
            "Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename)}",
        },
    )


@app.post("/ui/issues/{issue_id}/generate")
def issue_form_generate(issue_id: str, regenerate: bool = Form(False)) -> RedirectResponse:
    generate_issue(issue_id, regenerate)
    return RedirectResponse(url=f"/issues/{issue_id}", status_code=303)


@app.post("/ui/issues/{issue_id}/feedback")
def issue_form_feedback(
    issue_id: str,
    section_id: str = Form(...),
    original_text: str = Form(...),
    edited_text: str = Form(...),
    chosen_text: str = Form(""),
    rejected_text: str = Form(""),
    reason: str = Form(""),
    factuality: int = Form(3),
    style: int = Form(3),
    structure: int = Form(3),
    publishability: int = Form(3),
    verdict: str = Form("needs_edit"),
    notes: str = Form(""),
    flags: str = Form(""),
) -> RedirectResponse:
    from .services.feedback import record_feedback

    record_feedback(
        issue_id,
        FeedbackRequest(
            verdict=verdict,
            notes=notes,
            sections=[
                {
                    "section_id": section_id,
                    "original_text": original_text,
                    "edited_text": edited_text,
                    "chosen_text": chosen_text or edited_text,
                    "rejected_text": rejected_text or original_text,
                    "reason": reason or "manual_edit",
                    "factuality": factuality,
                    "style": style,
                    "structure": structure,
                    "publishability": publishability,
                    "flags": [item.strip() for item in flags.split(",") if item.strip()],
                    "notes": notes,
                }
            ],
        ),
    )
    return RedirectResponse(url=f"/issues/{issue_id}", status_code=303)


@app.post("/ui/training/sft")
def training_form_sft() -> RedirectResponse:
    run_sft_training()
    return RedirectResponse(url="/training", status_code=303)


@app.post("/ui/training/preferences")
def training_form_preferences() -> RedirectResponse:
    run_preference_training()
    return RedirectResponse(url="/training", status_code=303)


@app.post("/ui/eval/run")
def evaluation_form_run() -> RedirectResponse:
    run_evaluation()
    return RedirectResponse(url="/training", status_code=303)


@app.post("/ui/models/promote")
def promote_model_form(artifact_id: str = Form(...)) -> RedirectResponse:
    set_production_model(artifact_id)
    return RedirectResponse(url="/training", status_code=303)


@app.post("/ui/models/rollback")
def rollback_model_form(artifact_id: str = Form(...)) -> RedirectResponse:
    rollback_model(artifact_id)
    return RedirectResponse(url="/training", status_code=303)
