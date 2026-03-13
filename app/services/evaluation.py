from __future__ import annotations

import uuid
from statistics import mean

from ..db import get_connection
from ..utils import json_dumps, json_loads, now_iso


def run_evaluation() -> dict[str, object]:
    with get_connection() as connection:
        score_rows = connection.execute(
            """
            SELECT factuality, style, structure, publishability FROM section_scores ORDER BY created_at DESC LIMIT 100
            """
        ).fetchall()
        draft_rows = connection.execute(
            """
            SELECT metrics_json FROM draft_versions ORDER BY created_at DESC LIMIT 20
            """
        ).fetchall()
        production = connection.execute(
            """
            SELECT id FROM model_artifacts WHERE role = 'writer' AND is_production = 1 ORDER BY created_at DESC LIMIT 1
            """
        ).fetchone()

    publishability = [row["publishability"] for row in score_rows]
    style_scores = [row["style"] for row in score_rows]
    structure_scores = [row["structure"] for row in score_rows]
    factuality_scores = [row["factuality"] for row in score_rows]
    avg_length = mean(json_loads(row["metrics_json"], {}).get("approx_length", 0) for row in draft_rows) if draft_rows else 0
    summary = {
        "samples": len(score_rows),
        "avg_factuality": round(mean(factuality_scores), 2) if factuality_scores else 0,
        "avg_publishability": round(mean(publishability), 2) if publishability else 0,
        "avg_style": round(mean(style_scores), 2) if style_scores else 0,
        "avg_structure": round(mean(structure_scores), 2) if structure_scores else 0,
        "topic_selection_score": round(mean(publishability), 2) if publishability else 0,
        "voice_alignment_score": round(mean(style_scores), 2) if style_scores else 0,
        "insight_progression_score": round(mean(structure_scores), 2) if structure_scores else 0,
        "avg_approx_length": round(avg_length, 2) if avg_length else 0,
        "gate_passed": bool(
            publishability
            and mean(publishability) >= 4
            and style_scores
            and mean(style_scores) >= 4
            and structure_scores
            and mean(structure_scores) >= 4
        ),
    }
    eval_run_id = uuid.uuid4().hex
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO eval_runs (id, model_artifact_id, scope, status, summary_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                eval_run_id,
                production["id"] if production else None,
                "writer_regression",
                "completed",
                json_dumps(summary),
                now_iso(),
            ),
        )
    return {
        "eval_run_id": eval_run_id,
        "status": "completed",
        "summary": summary,
    }
