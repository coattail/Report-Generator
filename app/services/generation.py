from __future__ import annotations

import uuid
from typing import Any

from ..config import DEFAULT_MARKET_SYMBOLS
from ..db import get_connection
from ..utils import current_week_key, json_dumps, json_loads, now_iso
from .corpus import latest_style_profile
from .critic import build_citation_bundle, score_candidate
from .exports import export_issue, markdown_to_html
from .market import latest_market_snapshot, refresh_market_snapshot
from .runtime import LocalWriterRuntime
from .sources import CATEGORY_LABELS, get_topics, recommend_topics


def _topic_preference_text(topic: dict[str, Any]) -> str:
    category = CATEGORY_LABELS.get(topic.get("category", ""), topic.get("category", ""))
    summary = (topic.get("summary") or "").strip()
    event_summary = (topic.get("evidence_pack", {}) or {}).get("event_summary", "")
    details = summary or event_summary or topic.get("title", "")
    return f"{topic.get('title', '')}。分类：{category}。摘要：{details[:220]}".strip()


def _numbered_heading(index: int, title: str) -> str:
    return f"（{index}）{title}"


def _inline_links(citations: list[dict[str, Any]]) -> str:
    urls = [citation["url"] for citation in citations[:4] if citation.get("url")]
    if not urls:
        return "相关链接：暂无补充链接。"
    return "相关链接：" + "；".join(urls)


def _record_topic_selection_preferences(issue_id: str, week_key: str, topic_ids: list[str]) -> None:
    selected_ids = [topic_id for topic_id in topic_ids if topic_id]
    if not selected_ids:
        return

    topic_map = {topic["id"]: topic for topic in get_topics(week_key)}
    selected_topics = [topic_map[topic_id] for topic_id in selected_ids if topic_id in topic_map]
    if not selected_topics:
        return

    recommended_topics = recommend_topics(week_key, min_count=10, max_count=15)
    rejected_topics = [topic for topic in recommended_topics if topic["id"] not in set(selected_ids)]
    if not rejected_topics:
        return

    pairs: list[tuple[str, str, str]] = []
    for chosen_topic in selected_topics[:4]:
        chosen_text = _topic_preference_text(chosen_topic)
        for rejected_topic in rejected_topics[:3]:
            rejected_text = _topic_preference_text(rejected_topic)
            if chosen_text == rejected_text:
                continue
            pairs.append((chosen_text, rejected_text, "topic_selection_feedback"))
            if len(pairs) >= 12:
                break
        if len(pairs) >= 12:
            break

    if not pairs:
        return

    with get_connection() as connection:
        for chosen_text, rejected_text, reason in pairs:
            connection.execute(
                """
                INSERT INTO preference_pairs (
                  id, issue_id, section_id, chosen_text, rejected_text, reason, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uuid.uuid4().hex,
                    issue_id,
                    None,
                    chosen_text,
                    rejected_text,
                    reason,
                    now_iso(),
                ),
            )


def create_issue(week_key: str, title: str, topic_ids: list[str]) -> dict[str, Any]:
    issue_id = uuid.uuid4().hex
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO weekly_issues (
              id, week_key, title, selected_topic_ids_json, status, structure_json, market_scope_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                issue_id,
                week_key,
                title,
                json_dumps(topic_ids),
                "draft",
                json_dumps({}),
                json_dumps(list(DEFAULT_MARKET_SYMBOLS.keys())),
                now_iso(),
                now_iso(),
            ),
        )
    _record_topic_selection_preferences(issue_id, week_key, topic_ids)
    return get_issue(issue_id)


def get_issue(issue_id: str) -> dict[str, Any]:
    with get_connection() as connection:
        issue = connection.execute("SELECT * FROM weekly_issues WHERE id = ?", (issue_id,)).fetchone()
        sections = connection.execute(
            """
            SELECT * FROM issue_sections WHERE issue_id = ? ORDER BY created_at
            """,
            (issue_id,),
        ).fetchall()
        drafts = connection.execute(
            """
            SELECT * FROM draft_versions WHERE issue_id = ? ORDER BY created_at DESC
            """,
            (issue_id,),
        ).fetchall()
    if not issue:
        raise KeyError(issue_id)
    selected_topic_ids = json_loads(issue["selected_topic_ids_json"], [])
    topic_map = {topic["id"]: topic for topic in get_topics(issue["week_key"])}
    selected_topics = []
    for topic_id in selected_topic_ids:
        topic = topic_map.get(topic_id)
        if topic:
            selected_topics.append(
                {
                    "id": topic["id"],
                    "title": topic["title"],
                    "category": topic["category"],
                    "score": topic["score"],
                    "summary": topic["summary"],
                }
            )
        else:
            selected_topics.append(
                {
                    "id": topic_id,
                    "title": topic_id,
                    "category": "unresolved",
                    "score": 0,
                    "summary": "当前主题池已更新，原始选题未能在本周聚类中重新定位。",
                }
            )
    return {
        "id": issue["id"],
        "week_key": issue["week_key"],
        "title": issue["title"],
        "selected_topic_ids": selected_topic_ids,
        "selected_topics": selected_topics,
        "status": issue["status"],
        "structure": json_loads(issue["structure_json"], {}),
        "market_scope": json_loads(issue["market_scope_json"], []),
        "sections": [
            {
                "id": row["id"],
                "section_key": row["section_key"],
                "title": row["title"],
                "content": row["content"],
                "citations": json_loads(row["citations_json"], []),
                "candidate_rankings": json_loads(row["candidate_rankings_json"], []),
                "critic_notes": json_loads(row["critic_notes_json"], []),
                "final_score": row["final_score"],
            }
            for row in sections
        ],
        "drafts": [
            {
                "id": row["id"],
                "version_label": row["version_label"],
                "markdown_content": row["markdown_content"],
                "html_content": row["html_content"],
                "source_model_artifact_id": row["source_model_artifact_id"],
                "metrics": json_loads(row["metrics_json"], {}),
                "created_at": row["created_at"],
            }
            for row in drafts
        ],
    }


def list_issues() -> list[dict[str, Any]]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, week_key, title, status, created_at, updated_at
            FROM weekly_issues ORDER BY updated_at DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def generate_issue(issue_id: str, regenerate: bool = False, runtime_backend: str | None = None) -> dict[str, Any]:
    issue = get_issue(issue_id)
    if issue["drafts"] and not regenerate:
        return issue

    topics = [topic for topic in get_topics(issue["week_key"]) if topic["id"] in set(issue["selected_topic_ids"])]
    style_profile = latest_style_profile()
    runtime = LocalWriterRuntime(force_backend=runtime_backend)
    runtime_details = runtime.describe()
    structure = runtime.generate_outline(issue["title"], topics, style_profile)
    try:
        market_snapshot = refresh_market_snapshot(issue["week_key"])
    except Exception:
        market_snapshot = latest_market_snapshot(issue["week_key"])

    with get_connection() as connection:
        connection.execute("DELETE FROM issue_sections WHERE issue_id = ?", (issue_id,))

    section_records = []
    markdown_blocks = [f"# {issue['title']}"]
    if structure.get("lead"):
        markdown_blocks.append(structure["lead"])

    for topic_index, topic in enumerate(topics, start=1):
        candidates = runtime.generate_candidates(
            topic["title"],
            topic["evidence_pack"],
            style_profile,
            market_snapshot,
            topic.get("category"),
        )
        ranked = []
        for candidate in candidates:
            review = score_candidate(candidate["text"], topic["evidence_pack"], style_profile)
            ranked.append(
                {
                    "text": candidate["text"],
                    "score": review["score"],
                    "notes": review["notes"],
                    "sampling_note": candidate["sampling_note"],
                }
            )
        ranked.sort(key=lambda item: item["score"], reverse=True)
        chosen = ranked[0]
        citations = build_citation_bundle(topic["evidence_pack"])
        section_title = _numbered_heading(topic_index, topic["title"])
        section_records.append(
            {
                "id": uuid.uuid4().hex,
                "section_key": f"topic-{len(section_records) + 1}",
                "title": section_title,
                "content": chosen["text"],
                "citations": citations,
                "candidate_rankings": ranked,
                "critic_notes": chosen["notes"],
                "final_score": chosen["score"],
            }
        )
        markdown_blocks.append(section_title)
        markdown_blocks.append(_inline_links(citations))
        markdown_blocks.append("评述：")
        markdown_blocks.append(chosen["text"])

    market_text = runtime.generate_market_review(market_snapshot, style_profile)
    conclusion = runtime.generate_conclusion(topics, style_profile)
    market_title = _numbered_heading(len(topics) + 1, "市场评述")
    conclusion_title = _numbered_heading(len(topics) + 2, "本周总判断")
    section_records.extend(
        [
            {
                "id": uuid.uuid4().hex,
                "section_key": "market-review",
                "title": market_title,
                "content": market_text,
                "citations": [],
                "candidate_rankings": [],
                "critic_notes": [f"市场综述由 {runtime_details['backend']} 运行时结合本地行情快照生成。"],
                "final_score": 7.5,
            },
            {
                "id": uuid.uuid4().hex,
                "section_key": "conclusion",
                "title": conclusion_title,
                "content": conclusion,
                "citations": [],
                "candidate_rankings": [],
                "critic_notes": [f"结尾由 {runtime_details['backend']} 运行时对齐长期主义与估值框架。"],
                "final_score": 7.5,
            },
        ]
    )

    markdown_blocks.append(market_title)
    markdown_blocks.append(market_text)
    markdown_blocks.append(conclusion_title)
    markdown_blocks.append(conclusion)

    markdown_text = "\n\n".join(markdown_blocks)
    html_text = markdown_to_html(markdown_text)

    with get_connection() as connection:
        for section in section_records:
            connection.execute(
                """
                INSERT INTO issue_sections (
                  id, issue_id, section_key, title, content, citations_json,
                  candidate_rankings_json, critic_notes_json, final_score, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    section["id"],
                    issue_id,
                    section["section_key"],
                    section["title"],
                    section["content"],
                    json_dumps(section["citations"]),
                    json_dumps(section["candidate_rankings"]),
                    json_dumps(section["critic_notes"]),
                    section["final_score"],
                    now_iso(),
                    now_iso(),
                ),
            )
        draft_id = uuid.uuid4().hex
        connection.execute(
            """
            INSERT INTO draft_versions (
              id, issue_id, version_label, markdown_content, html_content, source_model_artifact_id, metrics_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                draft_id,
                issue_id,
                "v1" if not issue["drafts"] else f"v{len(issue['drafts']) + 1}",
                markdown_text,
                html_text,
                runtime_details["artifact_id"],
                json_dumps(
                    {
                        "section_count": len(section_records),
                        "topic_count": len(topics),
                        "target_length_range": [4500, 6500],
                        "approx_length": len(markdown_text),
                        "runtime": runtime_details,
                    }
                ),
                now_iso(),
            ),
        )
        connection.execute(
            """
            UPDATE weekly_issues SET status = ?, structure_json = ?, updated_at = ? WHERE id = ?
            """,
            ("generated", json_dumps(structure), now_iso(), issue_id),
        )

    export_issue(issue_id, markdown_text, html_text)
    return get_issue(issue_id)
