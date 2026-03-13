from __future__ import annotations

import difflib
import uuid
from typing import Any

from ..db import get_connection
from ..schemas import FeedbackRequest
from ..utils import json_dumps, now_iso
from .exports import export_issue, markdown_to_html


def record_feedback(issue_id: str, payload: FeedbackRequest) -> dict[str, Any]:
    with get_connection() as connection:
        issue = connection.execute("SELECT id FROM weekly_issues WHERE id = ?", (issue_id,)).fetchone()
        if not issue:
            raise KeyError(issue_id)

        for section in payload.sections:
            diff = list(
                difflib.unified_diff(
                    section.original_text.splitlines(),
                    section.edited_text.splitlines(),
                    lineterm="",
                )
            )
            connection.execute(
                """
                INSERT INTO edit_diffs (
                  id, issue_id, section_id, original_text, edited_text, diff_json, labels_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uuid.uuid4().hex,
                    issue_id,
                    section.section_id,
                    section.original_text,
                    section.edited_text,
                    json_dumps(diff),
                    json_dumps(section.flags),
                    now_iso(),
                ),
            )
            connection.execute(
                """
                INSERT INTO section_scores (
                  id, issue_id, section_id, factuality, style, structure, publishability, notes, flags_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uuid.uuid4().hex,
                    issue_id,
                    section.section_id,
                    section.factuality,
                    section.style,
                    section.structure,
                    section.publishability,
                    section.notes or "",
                    json_dumps(section.flags),
                    now_iso(),
                ),
            )
            if section.chosen_text and section.rejected_text:
                connection.execute(
                    """
                    INSERT INTO preference_pairs (
                      id, issue_id, section_id, chosen_text, rejected_text, reason, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        uuid.uuid4().hex,
                        issue_id,
                        section.section_id,
                        section.chosen_text,
                        section.rejected_text,
                        section.reason or "",
                        now_iso(),
                    ),
                )
            if section.section_id:
                connection.execute(
                    """
                    UPDATE issue_sections SET content = ?, updated_at = ? WHERE id = ?
                    """,
                    (section.edited_text, now_iso(), section.section_id),
                )

        connection.execute(
            """
            INSERT INTO publish_decisions (id, issue_id, verdict, notes, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                uuid.uuid4().hex,
                issue_id,
                payload.verdict,
                payload.notes or "",
                now_iso(),
            ),
        )
        connection.execute(
            """
            UPDATE weekly_issues SET status = ?, updated_at = ? WHERE id = ?
            """,
            ("reviewed" if payload.verdict == "publish_ready" else "needs_revision", now_iso(), issue_id),
        )

        section_rows = connection.execute(
            """
            SELECT title, content FROM issue_sections WHERE issue_id = ? ORDER BY created_at
            """,
            (issue_id,),
        ).fetchall()
        issue = connection.execute(
            """
            SELECT title FROM weekly_issues WHERE id = ?
            """,
            (issue_id,),
        ).fetchone()
        markdown_blocks = [f"# {issue['title']}"] if issue else ["# Weekly Issue"]
        for row in section_rows:
            markdown_blocks.append(f"## {row['title']}")
            markdown_blocks.append(row["content"])
        markdown_text = "\n\n".join(markdown_blocks)
        html_text = markdown_to_html(markdown_text)
        connection.execute(
            """
            INSERT INTO draft_versions (
              id, issue_id, version_label, markdown_content, html_content, source_model_artifact_id, metrics_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                uuid.uuid4().hex,
                issue_id,
                "feedback",
                markdown_text,
                html_text,
                None,
                json_dumps({"origin": "feedback_update", "section_count": len(section_rows), "approx_length": len(markdown_text)}),
                now_iso(),
            ),
        )

    export_issue(issue_id, markdown_text, html_text)
    return {"issue_id": issue_id, "verdict": payload.verdict, "sections_recorded": len(payload.sections)}
