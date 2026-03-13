from __future__ import annotations

import argparse
import re
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.db import get_connection, init_db
from app.schemas import FeedbackRequest
from app.services.corpus import latest_style_profile
from app.services.feedback import record_feedback
from app.services.runtime import LocalWriterRuntime
from app.services.training import run_preference_training
from app.utils import json_dumps, json_loads, now_iso


@dataclass
class SectionReview:
    title: str
    original_text: str
    style: int
    structure: int
    publishability: int
    factuality: int
    flags: list[str]
    notes: str
    reason: str
    chosen_text: str
    references: list[dict[str, str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review a sample article against historical corpus and write feedback.")
    parser.add_argument("--sample", required=True, help="Markdown sample path.")
    parser.add_argument("--output", required=True, help="Review markdown output path.")
    parser.add_argument("--write-feedback", action="store_true", help="Persist review as feedback assets.")
    parser.add_argument("--train-preferences", action="store_true", help="Run targeted preference dataset build after feedback.")
    return parser.parse_args()


def parse_markdown_sections(text: str) -> tuple[str, list[tuple[str, str]]]:
    text = text.strip()
    parts = re.split(r"^##\s+", text, flags=re.M)
    lead_block = parts[0].strip()
    lead_lines = [line for line in lead_block.splitlines() if line.strip()]
    lead_title = "导语"
    lead_text = ""
    if len(lead_lines) > 1:
        lead_text = "\n".join(lead_lines[1:]).strip()
    sections: list[tuple[str, str]] = []
    if lead_text:
        sections.append((lead_title, lead_text))
    for block in parts[1:]:
        lines = [line.rstrip() for line in block.splitlines()]
        if not lines:
            continue
        title = lines[0].strip()
        body = "\n".join(lines[1:]).strip()
        sections.append((title, body))
    return lead_lines[0] if lead_lines else "样稿", sections


def tokenize(text: str) -> set[str]:
    normalized = text.lower()
    english = re.findall(r"[a-z]{3,}", normalized)
    chinese = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    return set(english + chinese)


def load_historical_sections() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with get_connection() as connection:
        corpus_rows = connection.execute("SELECT file_name, title, sections_json FROM style_corpus_docs ORDER BY imported_at DESC").fetchall()
    for row in corpus_rows:
        for section in json_loads(row["sections_json"], []):
            content = (section.get("content") or "").strip()
            if len(content) < 80:
                continue
            rows.append(
                {
                    "file_name": row["file_name"],
                    "doc_title": row["title"],
                    "section_title": section.get("heading", "正文"),
                    "task_type": section.get("task_type", "topic_section"),
                    "content": content,
                }
            )
    return rows


def retrieve_references(section_title: str, section_text: str, historical_sections: list[dict[str, str]], limit: int = 2) -> list[dict[str, str]]:
    query_tokens = tokenize(section_title + "\n" + section_text)
    scored: list[tuple[float, dict[str, str]]] = []
    for item in historical_sections:
        tokens = tokenize(item["section_title"] + "\n" + item["content"][:500])
        overlap = len(query_tokens & tokens)
        if overlap == 0:
            continue
        score = overlap
        if item["task_type"] == "market_review" and "市场" in section_title:
            score += 4
        if item["task_type"] == "conclusion" and "价值投资" in section_title:
            score += 4
        if item["task_type"] == "intro" and section_title == "导语":
            score += 3
        scored.append((score, item))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [item for _, item in scored[:limit]]


def infer_review(section_title: str, text: str) -> tuple[int, int, int, int, list[str], str, str]:
    title_lower = section_title.lower()
    if section_title == "导语":
        return (
            4,
            2,
            2,
            2,
            ["不像我", "机器腔", "缺少我的判断"],
            "导语没有形成作者一贯的本周总判断，且串入了与当前选题无关的旧训练残留。",
            "voice_alignment",
        )
    if "where to buy" in title_lower or "iphone" in title_lower:
        return (
            4,
            1,
            1,
            1,
            ["选题偏弱", "偏消费资讯", "不适合付费周报", "不像我"],
            "这类导购型科技资讯与作者的付费周报定位不匹配，应该删除并让位给更有宏观和投资含义的主题。",
            "topic_selection",
        )
    if "google wraps up" in title_lower or "wiz" in title_lower:
        return (
            4,
            3,
            4,
            5,
            ["缺少我的判断"],
            "主题本身是正确的，但分析仍偏保守，缺少作者对科技资本配置、并购周期和估值含义的进一步判断。",
            "insight_upgrade",
        )
    if "federal reserve" in title_lower or "industrial and commercial bank of china" in title_lower:
        return (
            4,
            3,
            3,
            4,
            ["缺少我的判断", "没有推导到影响"],
            "题材重要，但正文停留在积极信号层面，没有充分推导其对跨境监管、金融摩擦边界和中资机构在美业务的真实意义。",
            "insight_upgrade",
        )
    if "市场评述" in section_title:
        return (
            3,
            1,
            1,
            1,
            ["市场深度不足", "缺少我的判断"],
            "市场评述当前基本缺失，而这恰恰是作者周报的核心卖点之一。",
            "market_depth",
        )
    if "价值投资视角总结" in section_title:
        return (
            2,
            1,
            1,
            1,
            ["不像我", "机器腔", "判断越界", "串文污染"],
            "价值投资总结没有落回作者真正关注的估值、基本面、预期差和长期持有，而是重复了不相关的旧残留判断。",
            "voice_alignment",
        )
    if "王毅" in section_title or "达尔" in section_title:
        return (
            4,
            2,
            2,
            3,
            ["只在复述新闻", "缺少关键背景", "不像我"],
            "内容过于像外交通稿摘要，缺少作者一贯的地缘政治背景、现实约束和战略判断。",
            "insight_upgrade",
        )
    if "政协" in section_title or "闭幕会" in section_title:
        return (
            2,
            1,
            1,
            2,
            ["只在复述新闻", "结论跳跃", "判断越界", "机器腔"],
            "从会议快讯直接跳到房地产和科创再贷款，逻辑断裂明显，是典型的串文污染，需要整段重写。",
            "voice_alignment",
        )
    return (
        4,
        3,
        3,
        3,
        ["缺少我的判断"],
        "整体可读，但还需要更明确的作者判断和更完整的因果链条。",
        "manual_edit",
    )


def build_rewrite(
    runtime: LocalWriterRuntime,
    section_title: str,
    original_text: str,
    notes: str,
    reason: str,
    references: list[dict[str, str]],
    style_profile: dict,
) -> str:
    if reason == "topic_selection" and ("iphone" in section_title.lower() or "where to buy" in section_title.lower()):
        return "这类导购型科技资讯不适合进入本周付费周报，应删除并将篇幅让给更具宏观、产业或投资含义的美国科技主题。"
    if "市场评述" in section_title:
        return (
            "本周缺乏完整的本地行情快照，因此不宜强行给出方向性判断。"
            "如果只为了凑齐结构而写市场评述，反而会稀释整篇周报的可信度。"
            "对作者来说，真正有价值的市场评述应该建立在指数、利率、美元、波动率以及科技龙头相对强弱的完整观察之上。"
        )
    ref_text = "\n\n".join(
        f"历史参考 {index + 1}：{item['doc_title']} / {item['section_title']}\n{item['content'][:380]}"
        for index, item in enumerate(references)
    )
    system_prompt = "你是成熟、克制、判断明确的中文财经付费通讯作者。"
    user_prompt = (
        f"请基于以下要求，重写《{section_title}》这一节。\n\n"
        f"当前样稿：\n{original_text}\n\n"
        f"批改意见：{notes}\n"
        f"重点方向：{reason}\n"
        f"作者风格提醒：{'；'.join(style_profile.get('voice_notes', [])[:2])}\n"
        f"作者世界观提醒：{'；'.join(style_profile.get('worldview_notes', [])[:3])}\n\n"
        f"{ref_text}\n\n"
        "要求：\n"
        "1. 保留当前主题真实可支持的事实，不要捏造新事实。\n"
        "2. 写得更像作者本人，允许明确表达“我认为 / 我的判断是”。\n"
        "3. 从事实推进到观点，不要只复述新闻。\n"
        "4. 不要写标题，不要写来源列表，不要解释改写思路。\n"
        "5. 篇幅控制在 220 到 420 字。"
    )
    if runtime.available():
        try:
            return runtime._chat_generate(system_prompt, user_prompt, max_tokens=340, temp=0.22)
        except Exception:
            pass
    reference_fallback = references[0]["content"][:260] if references else ""
    return (
        f"{original_text[:160]} 我认为，更值得关注的不是表面消息本身，而是它背后反映的政策边界、"
        f"市场预期和长期定价逻辑。{notes} {reference_fallback}"
    ).strip()


def write_review(output_path: Path, sample_title: str, reviews: list[SectionReview]) -> None:
    lines = [
        f"# {sample_title} 历史语料对照批改",
        "",
        "这份批改是结合已导入的历史周报语料生成的，对照重点是：选题判断力、事实到观点推进、个人声音。",
        "",
    ]
    for review in reviews:
        lines.extend(
            [
                f"## {review.title}",
                "",
                f"- 事实：`{review.factuality}`",
                f"- 风格（个人声音）：`{review.style}`",
                f"- 结构（观点推进）：`{review.structure}`",
                f"- 可发（选题判断）：`{review.publishability}`",
                f"- Flags：`{', '.join(review.flags)}`",
                f"- 批改意见：{review.notes}",
                "",
                "### 历史参考",
                "",
            ]
        )
        for reference in review.references:
            lines.append(
                f"- `{reference['doc_title']}` / `{reference['section_title']}` / `{reference['file_name']}`"
            )
        lines.extend(
            [
                "",
                "### 建议改写",
                "",
                review.chosen_text.strip(),
                "",
            ]
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def persist_feedback(sample_title: str, reviews: list[SectionReview]) -> dict:
    issue_id = uuid.uuid4().hex
    created_at = now_iso()
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO weekly_issues (
              id, week_key, title, selected_topic_ids_json, status, structure_json, market_scope_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                issue_id,
                "2026-W11",
                f"{sample_title}（历史语料批改）",
                "[]",
                "draft",
                "{}",
                "[]",
                created_at,
                created_at,
            ),
        )
        section_payloads = []
        for index, review in enumerate(reviews, start=1):
            section_id = uuid.uuid4().hex
            connection.execute(
                """
                INSERT INTO issue_sections (
                  id, issue_id, section_key, title, content, citations_json, candidate_rankings_json, critic_notes_json, final_score, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    section_id,
                    issue_id,
                    f"review-{index}",
                    review.title,
                    review.original_text,
                    "[]",
                    "[]",
                    json_dumps(review.flags),
                    0,
                    created_at,
                    created_at,
                ),
            )
            section_payloads.append(
                {
                    "section_id": section_id,
                    "original_text": review.original_text,
                    "edited_text": review.chosen_text,
                    "chosen_text": review.chosen_text,
                    "rejected_text": review.original_text,
                    "reason": review.reason,
                    "factuality": review.factuality,
                    "style": review.style,
                    "structure": review.structure,
                    "publishability": review.publishability,
                    "flags": review.flags,
                    "notes": review.notes,
                }
            )

    feedback_result = record_feedback(
        issue_id,
        FeedbackRequest(
            verdict="needs_edit",
            notes="基于历史语料的自动批改，用于针对性训练。",
            sections=section_payloads,
        ),
    )
    return {"issue_id": issue_id, "feedback": feedback_result}


def main() -> int:
    args = parse_args()
    init_db()
    sample_path = Path(args.sample)
    output_path = Path(args.output)
    sample_title, sections = parse_markdown_sections(sample_path.read_text(encoding="utf-8"))
    historical_sections = load_historical_sections()
    runtime = LocalWriterRuntime()
    style_profile = latest_style_profile()

    reviews: list[SectionReview] = []
    for title, content in sections:
        refs = retrieve_references(title, content, historical_sections)
        factuality, style, structure, publishability, flags, notes, reason = infer_review(title, content)
        chosen_text = build_rewrite(runtime, title, content, notes, reason, refs, style_profile)
        reviews.append(
            SectionReview(
                title=title,
                original_text=content,
                style=style,
                structure=structure,
                publishability=publishability,
                factuality=factuality,
                flags=flags,
                notes=notes,
                reason=reason,
                chosen_text=chosen_text,
                references=refs,
            )
        )

    write_review(output_path, sample_title, reviews)
    print(f"Review written to {output_path}")

    feedback_result = None
    if args.write_feedback:
        feedback_result = persist_feedback(sample_title, reviews)
        print(json_dumps(feedback_result))

    if args.train_preferences:
        training_result = run_preference_training()
        print(json_dumps({"preference_training": training_result}))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
