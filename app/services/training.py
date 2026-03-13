from __future__ import annotations

import random
import re
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from ..config import MODEL_DIR, TRAINING_DIR
from ..db import get_connection
from ..utils import json_dumps, json_loads, now_iso
from .corpus import latest_style_profile, resolved_sections

DEFAULT_WRITER_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"
FALLBACK_WRITER_MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"

TOPIC_SELECTION_FLAGS = {
    "选题偏弱",
    "不适合付费周报",
    "重要性不足",
    "偏消费资讯",
    "偏热闹不偏关键",
}
INSIGHT_FLAGS = {
    "只在复述新闻",
    "缺少关键背景",
    "没有推导到影响",
    "结论跳跃",
    "判断越界",
    "分析深度不足",
}
VOICE_FLAGS = {
    "不像我",
    "机器腔",
    "太圆滑",
    "空话偏多",
    "缺少我的判断",
    "价值投资味道不足",
}
TOPIC_REASON_TOKENS = ("topic", "selection", "publishability")
INSIGHT_REASON_TOKENS = ("insight", "depth", "analysis", "structure")
VOICE_REASON_TOKENS = ("voice", "tone", "style", "alignment")
WEAK_TOPIC_PATTERNS = [
    r"\bwhere to buy\b",
    r"\bhow to\b",
    r"\bpreorder\b",
    r"\bbuy\b",
    r"\bprice\b",
    r"\brelease date\b",
    r"\bdeals?\b",
    r"\bdeal\b",
    r"\bshopping\b",
    r"\bguide\b",
    r"\bhands-on\b",
    r"哪里买",
    r"怎么买",
    r"购买",
    r"预购",
    r"价格",
    r"开箱",
]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json_dumps(row) + "\n")


def _dedupe_examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    output: list[dict[str, Any]] = []
    for row in rows:
        key = (
            row.get("prompt", "").strip(),
            row.get("completion", "").strip(),
            row.get("focus_axis", "").strip(),
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output


def _brief_style_context() -> dict[str, str]:
    profile = latest_style_profile()
    voice_notes = "；".join(profile.get("voice_notes", [])[:2]) or "语气直接，强调判断。"
    worldview_notes = "；".join(profile.get("worldview_notes", [])[:2]) or "关注政策、市场、科技与长期价值。"
    banned = "、".join(profile.get("banned_phrases", [])[:3]) or "空话套话"
    blueprint = (profile.get("stats") or {}).get("blueprint") or {}
    structure = (blueprint.get("structure") or {}).get("topic_flow") or "主题标题 -> 相关链接 -> 评述"
    market_focuses = "、".join((blueprint.get("market_review") or {}).get("focuses", [])[:3]) or "valuation、rates、tech"
    return {
        "version": profile.get("version", "empty"),
        "voice_notes": voice_notes,
        "worldview_notes": worldview_notes,
        "banned_phrases": banned,
        "structure": structure,
        "market_focuses": market_focuses,
    }


def build_feedback_sft_examples(limit: int = 240) -> tuple[list[dict[str, Any]], dict[str, int]]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
              e.edited_text,
              e.original_text,
              COALESCE((
                SELECT p.chosen_text
                FROM preference_pairs p
                WHERE p.issue_id = e.issue_id
                  AND COALESCE(p.section_id, '') = COALESCE(e.section_id, '')
                ORDER BY p.created_at DESC
                LIMIT 1
              ), '') AS chosen_text,
              COALESCE((
                SELECT s.flags_json
                FROM section_scores s
                WHERE s.issue_id = e.issue_id
                  AND COALESCE(s.section_id, '') = COALESCE(e.section_id, '')
                ORDER BY s.created_at DESC
                LIMIT 1
              ), '[]') AS flags_json,
              COALESCE((
                SELECT s.notes
                FROM section_scores s
                WHERE s.issue_id = e.issue_id
                  AND COALESCE(s.section_id, '') = COALESCE(e.section_id, '')
                ORDER BY s.created_at DESC
                LIMIT 1
              ), '') AS notes,
              COALESCE((
                SELECT s.style
                FROM section_scores s
                WHERE s.issue_id = e.issue_id
                  AND COALESCE(s.section_id, '') = COALESCE(e.section_id, '')
                ORDER BY s.created_at DESC
                LIMIT 1
              ), 3) AS style,
              COALESCE((
                SELECT s.structure
                FROM section_scores s
                WHERE s.issue_id = e.issue_id
                  AND COALESCE(s.section_id, '') = COALESCE(e.section_id, '')
                ORDER BY s.created_at DESC
                LIMIT 1
              ), 3) AS structure,
              COALESCE((
                SELECT s.publishability
                FROM section_scores s
                WHERE s.issue_id = e.issue_id
                  AND COALESCE(s.section_id, '') = COALESCE(e.section_id, '')
                ORDER BY s.created_at DESC
                LIMIT 1
              ), 3) AS publishability,
              COALESCE(sec.title, '正文') AS section_title
            FROM edit_diffs e
            LEFT JOIN issue_sections sec ON sec.id = e.section_id
            ORDER BY e.created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    style_context = _brief_style_context()
    examples: list[dict[str, Any]] = []
    axis_counts: Counter[str] = Counter()
    for row in rows:
        chosen_text = (row["chosen_text"] or row["edited_text"] or "").strip()
        if not chosen_text or len(chosen_text) < 40:
            continue
        flags = json_loads(row["flags_json"], [])
        axes = _infer_preference_axes(
            "",
            flags,
            row["notes"] or "",
            int(row["style"] or 3),
            int(row["structure"] or 3),
            int(row["publishability"] or 3),
        )
        section_title = row["section_title"]
        for axis in axes:
            axis_counts[axis] += 1
            if axis == "topic_selection":
                prompt = (
                    f"请按作者的周报选题标准，重写《{section_title}》这一节。"
                    "要求：只保留真正值得纳入付费周报的内容，排除导购、消费资讯或重要性不足的展开；"
                    f"如果不该写，也要体现作者的取舍标准。风格取向：{style_context['worldview_notes']}。"
                )
            elif axis == "insight_progression":
                prompt = (
                    f"请重写《{section_title}》这一节。要求：先写关键事实，再解释为什么重要，"
                    "再推导到政策、行业、市场或资产影响，避免停留在新闻复述。"
                    f"写作取向：{style_context['worldview_notes']}。"
                )
            elif axis == "voice_alignment":
                prompt = (
                    f"请用作者本人会发布给付费读者的口吻，重写《{section_title}》这一节。"
                    "要求：有明确判断，可以使用“我认为 / 我觉得 / 我的判断是”，避免机器腔。"
                    f"风格提醒：{style_context['voice_notes']}。结构提醒：{style_context['structure']}。避免：{style_context['banned_phrases']}。"
                )
            else:
                prompt = f"请用作者风格重写《{section_title}》这一节。"
            examples.append(
                {
                    "task_type": "topic_section",
                    "focus_axis": axis,
                    "prompt": prompt,
                    "completion": chosen_text,
                    "title": section_title,
                }
            )
    axis_counts["total"] = len(examples)
    return examples, dict(axis_counts)


def build_targeted_sft_examples() -> tuple[list[dict[str, Any]], dict[str, int]]:
    with get_connection() as connection:
        rows = connection.execute("SELECT title, content, sections_json FROM style_corpus_docs ORDER BY imported_at DESC").fetchall()

    style_context = _brief_style_context()
    examples: list[dict[str, Any]] = []
    axis_counts: Counter[str] = Counter()
    for row in rows:
        sections = resolved_sections(row["content"], row["sections_json"])
        for section in sections:
            heading = (section.get("heading") or "正文").strip()
            content = (section.get("content") or "").strip()
            if not content:
                continue
            task_type = section.get("task_type", "topic_section")
            if task_type == "source_links":
                continue

            examples.append(
                {
                    "task_type": task_type,
                    "focus_axis": "baseline",
                    "prompt": f"请用作者的风格撰写《{row['title']}》中的{heading}部分。",
                    "completion": content,
                    "title": row["title"],
                }
            )
            axis_counts["baseline"] += 1

            if task_type in {"topic_section", "market_review", "conclusion"}:
                examples.append(
                    {
                        "task_type": task_type,
                        "focus_axis": "insight_progression",
                        "prompt": (
                            f"请模仿作者撰写《{row['title']}》中的{heading}部分。"
                            f"要求：先交代关键事实，再解释为什么重要，再给出作者判断，最后落到政策、行业、市场或资产层面的影响。"
                            f"写作特征：{style_context['worldview_notes']}。如果是市场评述，要重点看 {style_context['market_focuses']}。避免只复述新闻。"
                        ),
                        "completion": content,
                        "title": row["title"],
                    }
                )
                axis_counts["insight_progression"] += 1

            if task_type in {"intro", "topic_section", "market_review", "conclusion"}:
                examples.append(
                    {
                        "task_type": task_type,
                        "focus_axis": "voice_alignment",
                        "prompt": (
                            f"请以作者本人会发布给付费读者的口吻，重写《{row['title']}》中的{heading}部分。"
                            f"要求：可以明确使用“我认为 / 我觉得 / 我的判断是”，语气要有判断力但保留边界感。"
                            f"风格提醒：{style_context['voice_notes']}。结构提醒：{style_context['structure']}。避免：{style_context['banned_phrases']}。"
                        ),
                        "completion": content,
                        "title": row["title"],
                    }
                )
                axis_counts["voice_alignment"] += 1

            if task_type in {"intro", "conclusion"}:
                examples.append(
                    {
                        "task_type": task_type,
                        "focus_axis": "topic_selection",
                        "prompt": (
                            f"请按作者做周报选题时的标准，撰写《{row['title']}》中的{heading}部分。"
                            "要求：只突出真正值得写的主题，不要引入导购、消费资讯或热闹但不关键的题材；"
                            f"内容要体现作者对轻重缓急的取舍。价值取向：{style_context['worldview_notes']}。"
                        ),
                        "completion": content,
                        "title": row["title"],
                    }
                )
                axis_counts["topic_selection"] += 1

    deduped = _dedupe_examples(examples)
    feedback_examples, feedback_axis_counts = build_feedback_sft_examples()
    deduped = _dedupe_examples(deduped + feedback_examples)
    axis_counts["total"] = len(deduped)
    for key, value in feedback_axis_counts.items():
        axis_counts[f"feedback_{key}"] += value
    return deduped, dict(axis_counts)


def _looks_like_weak_topic(title: str) -> bool:
    normalized = title.lower().strip()
    return any(re.search(pattern, normalized) for pattern in WEAK_TOPIC_PATTERNS)


def _normalize_topic_heading(heading: str, content: str) -> str:
    normalized = re.sub(r"^[（(]\d+[）)]\s*", "", heading).strip("：: ")
    if normalized and normalized not in {"相关链接", "评述", "正文"}:
        return normalized
    first_sentence = content.split("。", 1)[0].split(".", 1)[0].strip()
    return first_sentence[:120] if first_sentence else heading


def build_topic_selection_pairs(limit: int = 240) -> list[dict[str, Any]]:
    with get_connection() as connection:
        corpus_rows = connection.execute("SELECT title, content, sections_json FROM style_corpus_docs ORDER BY imported_at DESC").fetchall()
        cluster_rows = connection.execute(
            "SELECT title, category, summary FROM topic_clusters ORDER BY created_at DESC LIMIT 300"
        ).fetchall()
        article_rows = connection.execute(
            "SELECT title, category, summary FROM raw_articles ORDER BY created_at DESC LIMIT 300"
        ).fetchall()

    chosen_topics: list[str] = []
    for row in corpus_rows:
        for section in resolved_sections(row["content"], row["sections_json"]):
            if section.get("task_type") != "topic_section":
                continue
            content = (section.get("content") or "").strip()
            if len(content) < 80:
                continue
            chosen_topics.append(_normalize_topic_heading(section.get("heading", "正文"), content))
    if not chosen_topics:
        chosen_topics = [
            row["title"]
            for row in list(cluster_rows) + list(article_rows)
            if row["title"] and not _looks_like_weak_topic(row["title"])
        ]

    rejected_topics = [
        row["title"]
        for row in list(cluster_rows) + list(article_rows)
        if row["title"] and _looks_like_weak_topic(row["title"])
    ]
    if not rejected_topics:
        rejected_topics = [
            "Where to buy the new iPhone",
            "最新手机购买指南",
            "新品开箱与价格汇总",
        ]

    pairs: list[dict[str, Any]] = []
    for index, chosen in enumerate(chosen_topics[:limit]):
        rejected = rejected_topics[index % len(rejected_topics)]
        pairs.append(
            {
                "prompt": "判断哪些主题适合进入付费财经周报，优先重要、可分析、能承载作者观点的主题，排除导购和消费资讯。",
                "chosen": chosen,
                "rejected": rejected,
                "reason": "topic_selection",
            }
        )
    return pairs


def _infer_preference_axes(
    reason: str,
    flags: list[str],
    notes: str,
    style: int,
    structure: int,
    publishability: int,
) -> list[str]:
    joined_reason = (reason or "").lower()
    joined_notes = notes or ""
    axes: list[str] = []
    if (
        publishability <= 3
        or any(flag in TOPIC_SELECTION_FLAGS for flag in flags)
        or any(token in joined_reason for token in TOPIC_REASON_TOKENS)
        or any(flag in joined_notes for flag in TOPIC_SELECTION_FLAGS)
    ):
        axes.append("topic_selection")
    if (
        structure <= 3
        or any(flag in INSIGHT_FLAGS for flag in flags)
        or any(token in joined_reason for token in INSIGHT_REASON_TOKENS)
        or any(flag in joined_notes for flag in INSIGHT_FLAGS)
    ):
        axes.append("insight_progression")
    if (
        style <= 3
        or any(flag in VOICE_FLAGS for flag in flags)
        or any(token in joined_reason for token in VOICE_REASON_TOKENS)
        or any(flag in joined_notes for flag in VOICE_FLAGS)
    ):
        axes.append("voice_alignment")
    return axes or ["general_preference"]


def build_targeted_preference_rows() -> tuple[list[dict[str, Any]], dict[str, int]]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
              p.chosen_text,
              p.rejected_text,
              p.reason,
              COALESCE((
                SELECT s.flags_json
                FROM section_scores s
                WHERE s.issue_id = p.issue_id
                  AND COALESCE(s.section_id, '') = COALESCE(p.section_id, '')
                ORDER BY s.created_at DESC
                LIMIT 1
              ), '[]') AS flags_json,
              COALESCE((
                SELECT s.notes
                FROM section_scores s
                WHERE s.issue_id = p.issue_id
                  AND COALESCE(s.section_id, '') = COALESCE(p.section_id, '')
                ORDER BY s.created_at DESC
                LIMIT 1
              ), '') AS notes,
              COALESCE((
                SELECT s.style
                FROM section_scores s
                WHERE s.issue_id = p.issue_id
                  AND COALESCE(s.section_id, '') = COALESCE(p.section_id, '')
                ORDER BY s.created_at DESC
                LIMIT 1
              ), 3) AS style,
              COALESCE((
                SELECT s.structure
                FROM section_scores s
                WHERE s.issue_id = p.issue_id
                  AND COALESCE(s.section_id, '') = COALESCE(p.section_id, '')
                ORDER BY s.created_at DESC
                LIMIT 1
              ), 3) AS structure,
              COALESCE((
                SELECT s.publishability
                FROM section_scores s
                WHERE s.issue_id = p.issue_id
                  AND COALESCE(s.section_id, '') = COALESCE(p.section_id, '')
                ORDER BY s.created_at DESC
                LIMIT 1
              ), 3) AS publishability
            FROM preference_pairs p
            ORDER BY p.created_at DESC
            """
        ).fetchall()

    dataset_rows: list[dict[str, Any]] = []
    axis_counts: Counter[str] = Counter()
    for row in rows:
        flags = json_loads(row["flags_json"], [])
        axes = _infer_preference_axes(
            row["reason"] or "",
            flags,
            row["notes"] or "",
            int(row["style"] or 3),
            int(row["structure"] or 3),
            int(row["publishability"] or 3),
        )
        for axis in axes:
            axis_counts[axis] += 1
        dataset_rows.append(
            {
                "chosen": row["chosen_text"],
                "rejected": row["rejected_text"],
                "reason": row["reason"] or "",
                "axes": axes,
                "flags": flags,
                "notes": row["notes"] or "",
            }
        )
    axis_counts["total"] = len(dataset_rows)
    return dataset_rows, dict(axis_counts)


def build_sft_splits(seed: int = 42) -> dict[str, list[dict[str, Any]]]:
    examples, _ = build_targeted_sft_examples()
    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)
    total = len(shuffled)
    if total < 10:
        return {
            "train": shuffled,
            "valid": shuffled[: max(1, total // 3)],
            "test": shuffled[: max(1, total // 3)],
        }

    valid_count = max(10, round(total * 0.1))
    test_count = max(10, round(total * 0.05))
    if valid_count + test_count >= total:
        valid_count = max(1, total // 5)
        test_count = max(1, total // 10)
    train_count = max(1, total - valid_count - test_count)
    train = shuffled[:train_count]
    valid = shuffled[train_count : train_count + valid_count]
    test = shuffled[train_count + valid_count :]
    return {"train": train, "valid": valid, "test": test}


def prepare_sft_dataset(dataset_dir: Path, seed: int = 42) -> dict[str, Any]:
    splits = build_sft_splits(seed=seed)
    for split_name, rows in splits.items():
        _write_jsonl(dataset_dir / f"{split_name}.jsonl", rows)
    _, axis_counts = build_targeted_sft_examples()
    return {
        "dataset_dir": str(dataset_dir),
        "counts": {split_name: len(rows) for split_name, rows in splits.items()},
        "example_count": sum(len(rows) for rows in splits.values()),
        "focus_axis_counts": axis_counts,
    }


def list_model_artifacts() -> list[dict[str, Any]]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT * FROM model_artifacts ORDER BY created_at DESC
            """
        ).fetchall()
    return [
        {
            "id": row["id"],
            "role": row["role"],
            "base_model": row["base_model"],
            "adapter_path": row["adapter_path"],
            "status": row["status"],
            "is_production": bool(row["is_production"]),
            "parent_artifact_id": row["parent_artifact_id"],
            "metrics": json_loads(row["metrics_json"], {}),
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def run_sft_training() -> dict[str, Any]:
    run_id = uuid.uuid4().hex
    dataset_path = TRAINING_DIR / "sft" / f"{run_id}.jsonl"
    examples, axis_counts = build_targeted_sft_examples()
    _write_jsonl(dataset_path, examples)
    topic_dataset_path = TRAINING_DIR / "topic_reranker" / f"{run_id}.jsonl"
    topic_pairs = build_topic_selection_pairs()
    _write_jsonl(topic_dataset_path, topic_pairs)
    artifact_id = uuid.uuid4().hex
    metrics = {
        "example_count": len(examples),
        "focus_axis_counts": axis_counts,
        "topic_selection_pair_count": len(topic_pairs),
        "topic_selection_dataset_path": str(topic_dataset_path),
        "style_profile_version": latest_style_profile()["version"],
        "runtime": "simulated-local",
        "next_step": "Install mlx-lm and replace simulated adapter execution with real LoRA fine-tuning.",
    }
    adapter_path = MODEL_DIR / "writer" / artifact_id
    adapter_path.mkdir(parents=True, exist_ok=True)
    (adapter_path / "README.txt").write_text(
        "This is a placeholder adapter directory. Wire this path to MLX LoRA outputs when the runtime is ready.\n",
        encoding="utf-8",
    )
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO model_artifacts (
              id, role, base_model, adapter_path, status, is_production, parent_artifact_id, metrics_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                artifact_id,
                "writer",
                DEFAULT_WRITER_MODEL,
                str(adapter_path),
                "trained",
                0,
                None,
                json_dumps(metrics),
                now_iso(),
            ),
        )
    return {
        "artifact_id": artifact_id,
        "role": "writer",
        "status": "trained",
        "dataset_path": str(dataset_path),
        "metrics": metrics,
    }


def run_preference_training() -> dict[str, Any]:
    dataset_rows, axis_counts = build_targeted_preference_rows()
    selection_feedback_pair_count = sum(1 for row in dataset_rows if row.get("reason") == "topic_selection_feedback")
    run_id = uuid.uuid4().hex
    dataset_path = TRAINING_DIR / "preferences" / f"{run_id}.jsonl"
    _write_jsonl(dataset_path, dataset_rows)
    topic_dataset_path = TRAINING_DIR / "topic_reranker" / f"{run_id}.jsonl"
    topic_pairs = build_topic_selection_pairs()
    _write_jsonl(topic_dataset_path, topic_pairs)
    artifact_id = uuid.uuid4().hex
    metrics = {
        "pair_count": len(dataset_rows),
        "selection_feedback_pair_count": selection_feedback_pair_count,
        "axis_counts": axis_counts,
        "topic_selection_pair_count": len(topic_pairs),
        "topic_selection_dataset_path": str(topic_dataset_path),
        "runtime": "simulated-local",
        "next_step": "Use pairwise ranking or DPO once MLX preference tooling is wired in.",
    }
    adapter_path = MODEL_DIR / "writer" / artifact_id
    adapter_path.mkdir(parents=True, exist_ok=True)
    (adapter_path / "README.txt").write_text(
        "Preference-tuned placeholder adapter. Replace with real training output once local preference optimization is configured.\n",
        encoding="utf-8",
    )
    with get_connection() as connection:
        latest_production = connection.execute(
            """
            SELECT id FROM model_artifacts WHERE role = 'writer' AND is_production = 1 ORDER BY created_at DESC LIMIT 1
            """
        ).fetchone()
        connection.execute(
            """
            INSERT INTO model_artifacts (
              id, role, base_model, adapter_path, status, is_production, parent_artifact_id, metrics_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                artifact_id,
                "writer",
                DEFAULT_WRITER_MODEL,
                str(adapter_path),
                "trained",
                0,
                latest_production["id"] if latest_production else None,
                json_dumps(metrics),
                now_iso(),
            ),
        )
    return {
        "artifact_id": artifact_id,
        "role": "writer",
        "status": "trained",
        "dataset_path": str(dataset_path),
        "metrics": metrics,
    }


def set_production_model(artifact_id: str) -> dict[str, Any]:
    with get_connection() as connection:
        target = connection.execute("SELECT * FROM model_artifacts WHERE id = ?", (artifact_id,)).fetchone()
        if not target:
            raise KeyError(artifact_id)
        connection.execute("UPDATE model_artifacts SET is_production = 0 WHERE role = ?", (target["role"],))
        connection.execute("UPDATE model_artifacts SET is_production = 1, status = ? WHERE id = ?", ("production", artifact_id))
    return get_production_model(target["role"])


def rollback_model(artifact_id: str) -> dict[str, Any]:
    with get_connection() as connection:
        artifact = connection.execute("SELECT * FROM model_artifacts WHERE id = ?", (artifact_id,)).fetchone()
        if not artifact:
            raise KeyError(artifact_id)
        parent_id = artifact["parent_artifact_id"]
        if not parent_id:
            raise ValueError("Artifact has no parent to roll back to.")
    return set_production_model(parent_id)


def get_production_model(role: str = "writer") -> dict[str, Any]:
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT * FROM model_artifacts WHERE role = ? AND is_production = 1 ORDER BY created_at DESC LIMIT 1
            """,
            (role,),
        ).fetchone()
    if not row:
        return {
            "id": None,
            "role": role,
            "status": "not_set",
        }
    return {
        "id": row["id"],
        "role": row["role"],
        "base_model": row["base_model"],
        "adapter_path": row["adapter_path"],
        "status": row["status"],
        "metrics": json_loads(row["metrics_json"], {}),
        "created_at": row["created_at"],
    }
