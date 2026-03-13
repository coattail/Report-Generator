from __future__ import annotations

import os
import re
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from docx import Document
from fastapi import UploadFile
from pypdf import PdfReader

from ..config import UPLOAD_DIR
from ..db import get_connection
from ..utils import compact_whitespace, hash_text, json_dumps, json_loads, now_iso, paragraph_split, sentence_split


CORPUS_UPLOAD_DIR = UPLOAD_DIR / "corpus"
PDF_PASSWORD_ENV = "BRIEFING_PDF_PASSWORD"
STANCE_MARKERS = ("我认为", "我觉得", "我的判断", "这意味着", "关键是", "更重要的是", "真正值得关注")
WORLDVIEW_MARKERS = (
    "估值",
    "长期",
    "现金流",
    "利率",
    "政策",
    "科技",
    "周期",
    "预期差",
    "风险偏好",
    "地缘政治",
    "中美",
    "台海",
    "俄乌",
    "伊朗",
)
MARKET_KEYWORDS = {
    "valuation": ("估值", "现金流", "盈利", "定价", "风险溢价"),
    "rates": ("利率", "通胀", "降息", "加息", "美债", "收益率", "Fed"),
    "tech": ("科技", "AI", "芯片", "云", "capex", "纳指", "大盘科技"),
    "macro": ("周期", "政策", "美元", "风险偏好", "流动性", "财政"),
}


def _save_upload(file_name: str, content: bytes) -> Path:
    CORPUS_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = f"{uuid.uuid4().hex[:8]}-{Path(file_name).name}"
    file_path = CORPUS_UPLOAD_DIR / safe_name
    file_path.write_bytes(content)
    return file_path


def _extract_docx(file_path: Path) -> str:
    document = Document(str(file_path))
    parts = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
    return compact_whitespace("\n\n".join(parts))


def _resolve_pdf_password(explicit_password: str | None = None) -> str | None:
    if explicit_password:
        return explicit_password
    password = os.getenv(PDF_PASSWORD_ENV, "").strip()
    return password or None


def _is_section_marker(line: str) -> bool:
    normalized = line.strip()
    return bool(
        normalized
        and (
            re.match(r"^每周订阅内容\s*\d", normalized)
            or re.match(r"^[（(]\d+[）)]", normalized)
            or normalized.startswith(("相关链接", "评述", "市场评述", "价值投资视角总结", "本周总判断"))
        )
    )


def _normalize_heading(heading: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(heading or "")).strip()
    cleaned = cleaned.replace("：", ":")
    cleaned = re.sub(r"^[#*\-]+\s*", "", cleaned)
    return cleaned.strip()


def _strip_section_prefix(text: str) -> str:
    cleaned = _normalize_heading(text)
    cleaned = re.sub(r"^每周订阅内容\s*\d+\s*", "", cleaned)
    cleaned = re.sub(r"^[（(]\d+[）)]\s*", "", cleaned)
    cleaned = re.sub(r"^(相关链接|评述|市场评述|价值投资视角总结|本周总判断)\s*[:：]?\s*", "", cleaned)
    return cleaned.strip()


def _looks_like_noise(text: str) -> bool:
    cleaned = _normalize_heading(text)
    if not cleaned:
        return True
    lowered = cleaned.lower()
    if "http://" in lowered or "https://" in lowered:
        return True
    if "每周订阅内容" in cleaned:
        return True
    digit_count = sum(ch.isdigit() for ch in cleaned)
    if digit_count > max(4, len(cleaned) // 4):
        return True
    return False


def _clean_profile_line(text: str, limit: int = 88) -> str:
    cleaned = _strip_section_prefix(text)
    cleaned = re.sub(r"(相关链接|评述|市场评述|本周总判断)\s*[:：]?", " ", cleaned)
    cleaned = re.sub(r"https?://\S+", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip("：: ;；,.，。 ")
    if len(cleaned) > limit:
        cleaned = cleaned[:limit].rstrip("，。；; ") + "..."
    return cleaned


def _topic_heading_label(heading: str, content: str) -> str:
    normalized = _strip_section_prefix(heading)
    if normalized and normalized not in {"正文", "市场评述"}:
        return normalized
    first_sentence = sentence_split(content[:160])
    return _clean_profile_line(first_sentence[0] if first_sentence else content[:40], limit=48) or "主题"


def _structure_blueprint(all_sections: list[dict[str, Any]]) -> dict[str, Any]:
    docs = defaultdict(list)
    for section in all_sections:
        doc_id = str(section.get("_doc_id", ""))
        docs[doc_id].append(section)

    link_docs = 0
    commentary_docs = 0
    market_docs = 0
    conclusion_docs = 0
    numbered_topics = 0
    topic_title_samples: list[str] = []

    for sections in docs.values():
        normalized_headings = [_normalize_heading(section.get("heading", "")) for section in sections]
        if any(heading.startswith("相关链接") for heading in normalized_headings):
            link_docs += 1
        if any(heading.startswith("评述") for heading in normalized_headings):
            commentary_docs += 1
        if any("市场评述" in heading for heading in normalized_headings):
            market_docs += 1
        if any(heading in {"价值投资视角总结", "本周总判断"} for heading in normalized_headings):
            conclusion_docs += 1
        numbered_topics += sum(1 for heading in normalized_headings if re.match(r"^[（(]\d+[）)]", heading))
        for section in sections:
            if section.get("task_type") == "topic_section":
                label = _topic_heading_label(section.get("heading", ""), section.get("content", ""))
                if label and label not in topic_title_samples:
                    topic_title_samples.append(label)
                if len(topic_title_samples) >= 5:
                    break

    doc_count = max(len(docs), 1)
    topic_flow = "主题标题 -> 相关链接 -> 评述" if link_docs >= max(1, doc_count // 3) else "主题标题 -> 评述"
    return {
        "doc_count": len(docs),
        "topic_flow": topic_flow,
        "uses_inline_links_ratio": round(link_docs / doc_count, 2),
        "uses_commentary_ratio": round(commentary_docs / doc_count, 2),
        "uses_market_review_ratio": round(market_docs / doc_count, 2),
        "uses_conclusion_ratio": round(conclusion_docs / doc_count, 2),
        "numbered_topic_titles": numbered_topics,
        "topic_title_samples": topic_title_samples[:5],
    }


def _market_review_profile(rows: list[Any], all_sections: list[dict[str, Any]]) -> dict[str, Any]:
    market_sentences: list[str] = []
    focus_counter: Counter[str] = Counter()
    for row in rows:
        for sentence in sentence_split(row["content"]):
            if any(keyword in sentence for keyword_group in MARKET_KEYWORDS.values() for keyword in keyword_group):
                market_sentences.append(sentence)
                for focus_key, keywords in MARKET_KEYWORDS.items():
                    if any(keyword in sentence for keyword in keywords):
                        focus_counter[focus_key] += 1

    for section in all_sections:
        if section.get("task_type") != "market_review":
            continue
        for sentence in sentence_split(section.get("content", "")):
            market_sentences.append(sentence)
            for focus_key, keywords in MARKET_KEYWORDS.items():
                if any(keyword in sentence for keyword in keywords):
                    focus_counter[focus_key] += 1

    cleaned_examples: list[str] = []
    for sentence in market_sentences:
        cleaned = _clean_profile_line(sentence)
        if len(cleaned) < 14 or _looks_like_noise(cleaned):
            continue
        if cleaned not in cleaned_examples:
            cleaned_examples.append(cleaned)
        if len(cleaned_examples) >= 4:
            break

    ordered_focus = [key for key, _ in focus_counter.most_common(4)]
    return {
        "focuses": ordered_focus or ["valuation", "rates", "tech"],
        "examples": cleaned_examples,
    }


def _voice_and_worldview_profile(texts: list[str], all_sections: list[dict[str, Any]]) -> tuple[list[str], list[str], dict[str, Any]]:
    sentences = [sentence for text in texts for sentence in sentence_split(text)]
    paragraphs = [paragraph for text in texts for paragraph in paragraph_split(text)]
    stance_counter: Counter[str] = Counter()
    marker_counter: Counter[str] = Counter()
    worldview_theme_counter: Counter[str] = Counter()

    for sentence in sentences:
        cleaned = _clean_profile_line(sentence)
        if len(cleaned) < 10 or _looks_like_noise(cleaned):
            continue
        for marker in STANCE_MARKERS:
            if marker in sentence:
                marker_counter[marker] += 1
                stance_counter[cleaned] += 2
        if any(token in sentence for token in ("估值", "现金流", "长期", "预期差")):
            worldview_theme_counter["valuation"] += 1
        if any(token in sentence for token in ("利率", "通胀", "美元", "流动性", "美债")):
            worldview_theme_counter["macro_rates"] += 1
        if any(token in sentence for token in ("科技", "AI", "芯片", "云", "半导体")):
            worldview_theme_counter["technology"] += 1
        if any(token in sentence for token in ("政策", "中美", "台海", "俄乌", "伊朗", "地缘政治")):
            worldview_theme_counter["policy_geo"] += 1

    avg_paragraph_length = round(sum(len(p) for p in paragraphs) / max(len(paragraphs), 1))
    avg_sentence_length = round(sum(len(s) for s in sentences) / max(len(sentences), 1))
    topic_sections = sum(1 for section in all_sections if section.get("task_type") == "topic_section")
    explicit_stance_ratio = round(sum(marker_counter.values()) / max(len(sentences), 1), 3)

    dominant_markers = [marker for marker, _ in marker_counter.most_common(3)]
    voice_notes = [
        "语气应直接下判断，但判断要紧贴事实边界，避免空泛平衡式表达。",
        f"常见作者口吻会使用 {' / '.join(dominant_markers) if dominant_markers else '“我认为 / 这意味着 / 更重要的是”'} 这类转折，先讲事实再点判断。",
        f"平均段落长度约 {avg_paragraph_length} 字，平均句长约 {avg_sentence_length} 字，正文通常不是短促快讯，而是完整分析段落。",
        f"历史语料里共识最强的是 {topic_sections} 个主题段落都强调“事实 -> 解释 -> 判断 -> 影响”这一推进方式。",
    ]

    worldview_notes: list[str] = []
    ordered_themes = [theme for theme, _ in worldview_theme_counter.most_common()]
    if "valuation" in ordered_themes:
        worldview_notes.append("作者更关注估值锚、现金流质量和中长期预期差，而不是一周情绪波动。")
    if "macro_rates" in ordered_themes:
        worldview_notes.append("遇到宏观话题时，会优先追问利率路径、通胀方向和美元流动性，而不是停留在数据表面。")
    if "technology" in ordered_themes:
        worldview_notes.append("科技议题不会被当成单纯产业新闻，而会被放回资本开支、竞争格局和资产定价框架里理解。")
    if "policy_geo" in ordered_themes:
        worldview_notes.append("国际政治与政策议题的落点通常会回到政策边界、地缘风险和市场风险偏好的二阶影响。")
    if not worldview_notes:
        worldview_notes = [
            "作者更关注政策边界、估值锚和中长期预期差，而不是一周的情绪波动。",
            "科技、宏观与地缘政治会被放在同一个资产定价框架里理解。",
        ]
    worldview_notes = worldview_notes[:4]

    return voice_notes[:4], worldview_notes, {"explicit_stance_ratio": explicit_stance_ratio, "dominant_markers": dominant_markers}


def _join_pdf_lines(lines: list[str]) -> str:
    paragraphs: list[str] = []
    bucket = ""

    def flush() -> None:
        nonlocal bucket
        if bucket.strip():
            paragraphs.append(bucket.strip())
        bucket = ""

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            flush()
            continue
        if _is_section_marker(line):
            flush()
            paragraphs.append(line)
            continue
        if not bucket:
            bucket = line
            continue
        if bucket.endswith("-"):
            bucket = f"{bucket[:-1]}{line}"
            continue
        if bucket.endswith(("。", "！", "？", "!", "?", "：", ":")):
            flush()
            bucket = line
            continue
        bucket = f"{bucket}{line}"

    flush()
    return "\n\n".join(paragraphs)


def _extract_pdf(file_path: Path, password: str | None = None) -> str:
    reader = PdfReader(str(file_path), strict=False)
    resolved_password = _resolve_pdf_password(password)
    if reader.is_encrypted:
        if not resolved_password:
            raise ValueError(f"Encrypted PDF requires a password: {file_path.name}")
        decrypt_result = reader.decrypt(resolved_password)
        if not decrypt_result:
            raise ValueError(f"Failed to decrypt PDF: {file_path.name}")
    page_blocks: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        lines = [line for line in page_text.replace("\r", "\n").split("\n")]
        page_blocks.append(_join_pdf_lines(lines))
    return compact_whitespace("\n\n".join(block for block in page_blocks if block.strip()))


def _extract_txt(file_path: Path) -> str:
    return compact_whitespace(file_path.read_text(encoding="utf-8", errors="ignore"))


def extract_text(file_path: Path, pdf_password: str | None = None) -> str:
    suffix = file_path.suffix.lower()
    if suffix == ".docx":
        return _extract_docx(file_path)
    if suffix == ".pdf":
        return _extract_pdf(file_path, pdf_password)
    return _extract_txt(file_path)


def infer_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        cleaned = line.strip(" #-*")
        if 4 <= len(cleaned) <= 80:
            return cleaned
    return Path(fallback).stem


def split_sections(text: str) -> list[dict[str, Any]]:
    paragraphs = paragraph_split(text)
    sections: list[dict[str, Any]] = []
    current_heading = "正文"
    current_bucket: list[str] = []
    active_topic_heading: str | None = None

    def flush() -> None:
        nonlocal current_bucket
        if current_bucket:
            content = "\n\n".join(current_bucket).strip()
            sections.append(
                {
                    "heading": current_heading,
                    "content": content,
                    "task_type": classify_section(current_heading, content, len(sections)),
                }
            )
            current_bucket = []

    for paragraph in paragraphs:
        normalized = paragraph.replace("：", ":").strip()
        looks_like_heading = (
            _is_section_marker(normalized)
            or (
                len(normalized) <= 36
                and not normalized.endswith(("。", ".", "？", "?", "！", "!", "：", ":"))
                and len(normalized.split()) <= 6
            )
        )
        if re.match(r"^[（(]\d+[）)]", normalized):
            flush()
            current_heading = normalized
            active_topic_heading = normalized
            continue
        if normalized.startswith("相关链接"):
            flush()
            link_content = re.sub(r"^相关链接\s*[:：]?\s*", "", paragraph).strip()
            sections.append(
                {
                    "heading": "相关链接",
                    "content": link_content or paragraph.strip(),
                    "task_type": "source_links",
                }
            )
            if active_topic_heading:
                current_heading = active_topic_heading
            continue
        if normalized.startswith("评述"):
            flush()
            current_heading = active_topic_heading or normalized
            continue
        if looks_like_heading:
            flush()
            current_heading = normalized
            if normalized in {"市场评述", "价值投资视角总结", "本周总判断"}:
                active_topic_heading = None
        else:
            current_bucket.append(paragraph)
    flush()

    if not sections and text.strip():
        sections.append(
            {
                "heading": "正文",
                "content": text.strip(),
                "task_type": "topic_section",
            }
        )
    return sections


def classify_section(heading: str, content: str, position: int) -> str:
    normalized_heading = _normalize_heading(heading)
    joined = f"{normalized_heading} {content[:160]}".lower()
    if normalized_heading.startswith("相关链接"):
        return "source_links"
    if normalized_heading.startswith("评述"):
        return "commentary"
    if "市场评述" in normalized_heading or any(
        token in joined for token in ["市场", "美股", "纳指", "标普", "market", "valuation", "估值"]
    ):
        return "market_review"
    if normalized_heading in {"价值投资视角总结", "本周总判断"} or any(
        token in joined for token in ["总结", "结语", "最后", "conclusion"]
    ):
        return "conclusion"
    if position == 0 or any(token in normalized_heading.lower() for token in ["导语", "开篇", "lead", "opening"]):
        return "intro"
    return "topic_section"


def derive_structure_tags(text: str) -> list[str]:
    tags: list[str] = []
    if "长期" in text or "long-term" in text.lower():
        tags.append("long_term")
    if any(token in text for token in ["估值", "valuation"]):
        tags.append("valuation")
    if any(token in text for token in ["利率", "inflation", "通胀"]):
        tags.append("macro_rates")
    if any(token in text for token in ["科技", "AI", "芯片", "cloud"]):
        tags.append("technology")
    return sorted(set(tags))


def build_style_profile() -> dict[str, Any]:
    with get_connection() as connection:
        rows = connection.execute("SELECT title, content, sections_json FROM style_corpus_docs ORDER BY imported_at DESC").fetchall()

        if not rows:
            profile = {
                "version": "empty",
                "voice_notes": ["尚未导入历史语料。"],
                "worldview_notes": [],
                "banned_phrases": [],
                "preferred_patterns": [],
                "stats": {
                    "blueprint": {
                        "structure": {"topic_flow": "主题标题 -> 相关链接 -> 评述"},
                        "market_review": {"focuses": ["valuation", "rates", "tech"], "examples": []},
                        "voice": {"explicit_stance_ratio": 0, "dominant_markers": []},
                    }
                },
            }
        else:
            texts = [row["content"] for row in rows]
            all_sections = []
            for row in rows:
                for section in resolved_sections(row["content"], row["sections_json"]):
                    normalized_section = dict(section)
                    normalized_section["_doc_id"] = row["title"]
                    all_sections.append(normalized_section)
            structure = _structure_blueprint(all_sections)
            market_profile = _market_review_profile(rows, all_sections)
            voice_notes, worldview_notes, voice_stats = _voice_and_worldview_profile(texts, all_sections)
            preferred = [
                f"主体结构优先采用“{structure['topic_flow']}”，每个主题单独成段。",
                "市场评述放在主题之后，重点讨论估值、利率路径、风险偏好和科技主线。",
                "结尾需要回到本周总判断，明确哪些变量值得继续跟踪，哪些噪音不必过度反应。",
            ]
            profile = {
                "version": now_iso(),
                "voice_notes": voice_notes,
                "worldview_notes": worldview_notes,
                "banned_phrases": ["稳赚不赔", "绝对正确", "毫无风险"],
                "preferred_patterns": preferred,
                "stats": {
                    "document_count": len(rows),
                    "section_count": len(all_sections),
                    "sentence_count": sum(len(sentence_split(text)) for text in texts),
                    "blueprint": {
                        "structure": structure,
                        "market_review": market_profile,
                        "voice": voice_stats,
                    },
                },
            }

        connection.execute(
            """
            INSERT INTO style_profiles (
              id, version, voice_notes_json, worldview_notes_json, banned_phrases_json,
              preferred_patterns_json, stats_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                uuid.uuid4().hex,
                profile["version"],
                json_dumps(profile["voice_notes"]),
                json_dumps(profile["worldview_notes"]),
                json_dumps(profile["banned_phrases"]),
                json_dumps(profile["preferred_patterns"]),
                json_dumps(profile["stats"]),
                now_iso(),
            ),
        )
        return profile


def _import_document(file_path: Path, display_name: str, pdf_password: str | None = None) -> bool:
    raw_text = extract_text(file_path, pdf_password)
    cleaned_text = compact_whitespace(raw_text)
    if not cleaned_text:
        return False
    metadata = {
        "content_hash": hash_text(cleaned_text),
        "paragraph_count": len(paragraph_split(cleaned_text)),
        "sentence_count": len(sentence_split(cleaned_text)),
    }
    title = infer_title(cleaned_text, display_name)
    sections = split_sections(cleaned_text)
    with get_connection() as connection:
        existing = connection.execute(
            """
            SELECT id FROM style_corpus_docs
            WHERE file_name = ? OR json_extract(metadata_json, '$.content_hash') = ?
            LIMIT 1
            """,
            (display_name, metadata["content_hash"]),
        ).fetchone()
        if existing:
            return False
        connection.execute(
            """
            INSERT INTO style_corpus_docs (
              id, file_name, file_type, imported_at, title, content,
              sections_json, metadata_json, structure_tags_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                uuid.uuid4().hex,
                display_name,
                file_path.suffix.lower().lstrip(".") or "txt",
                now_iso(),
                title,
                cleaned_text,
                json_dumps(sections),
                json_dumps(metadata),
                json_dumps(derive_structure_tags(cleaned_text)),
            ),
        )
    return True


async def import_uploads(files: list[UploadFile], pdf_password: str | None = None) -> dict[str, Any]:
    imported = 0
    skipped = 0
    for upload in files:
        content = await upload.read()
        file_path = _save_upload(upload.filename or f"upload-{uuid.uuid4().hex}.txt", content)
        if _import_document(file_path, upload.filename or file_path.name, pdf_password):
            imported += 1
        else:
            skipped += 1

    profile = build_style_profile()
    return {
        "imported_documents": imported,
        "skipped_documents": skipped,
        "style_profile_version": profile["version"],
    }


def import_local_paths(paths: list[Path], pdf_password: str | None = None) -> dict[str, Any]:
    imported = 0
    skipped = 0
    for path in paths:
        if _import_document(path, path.name, pdf_password):
            imported += 1
        else:
            skipped += 1

    profile = build_style_profile()
    return {
        "imported_documents": imported,
        "skipped_documents": skipped,
        "style_profile_version": profile["version"],
    }


def latest_style_profile() -> dict[str, Any]:
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT version, voice_notes_json, worldview_notes_json, banned_phrases_json,
                   preferred_patterns_json, stats_json
            FROM style_profiles ORDER BY created_at DESC LIMIT 1
            """
        ).fetchone()
    if not row:
        return {
            "version": "empty",
            "voice_notes": [],
            "worldview_notes": [],
            "banned_phrases": [],
            "preferred_patterns": [],
            "stats": {
                "blueprint": {
                    "structure": {"topic_flow": "主题标题 -> 相关链接 -> 评述"},
                    "market_review": {"focuses": ["valuation", "rates", "tech"], "examples": []},
                    "voice": {"explicit_stance_ratio": 0, "dominant_markers": []},
                }
            },
        }
    return {
        "version": row["version"],
        "voice_notes": json_loads(row["voice_notes_json"], []),
        "worldview_notes": json_loads(row["worldview_notes_json"], []),
        "banned_phrases": json_loads(row["banned_phrases_json"], []),
        "preferred_patterns": json_loads(row["preferred_patterns_json"], []),
        "stats": json_loads(row["stats_json"], {}),
    }


def resolved_sections(content: str, stored_sections: Any) -> list[dict[str, Any]]:
    parsed_sections = split_sections(content)
    normalized_parsed = [section for section in parsed_sections if section.get("content")]
    normalized_stored = [section for section in json_loads(stored_sections, []) if section.get("content")]

    def score(sections: list[dict[str, Any]]) -> tuple[int, int]:
        topic_like = sum(
            1
            for section in sections
            if re.match(r"^[（(]\d+[）)]", str(section.get("heading", ""))) or section.get("task_type") == "topic_section"
        )
        commentary_like = sum(
            1 for section in sections if str(section.get("heading", "")).startswith(("评述", "市场评述", "本周总判断"))
        )
        return topic_like + commentary_like, len(sections)

    return normalized_parsed if score(normalized_parsed) >= score(normalized_stored) else normalized_stored


def corpus_overview() -> dict[str, Any]:
    with get_connection() as connection:
        count_row = connection.execute("SELECT COUNT(*) AS count FROM style_corpus_docs").fetchone()
        docs = connection.execute(
            "SELECT file_name, title, imported_at, structure_tags_json FROM style_corpus_docs ORDER BY imported_at DESC LIMIT 12"
        ).fetchall()

    return {
        "count": count_row["count"] if count_row else 0,
        "documents": [
            {
                "file_name": row["file_name"],
                "title": row["title"],
                "imported_at": row["imported_at"],
                "structure_tags": json_loads(row["structure_tags_json"], []),
            }
            for row in docs
        ],
        "profile": latest_style_profile(),
    }


def build_training_examples() -> list[dict[str, Any]]:
    with get_connection() as connection:
        rows = connection.execute("SELECT title, content, sections_json FROM style_corpus_docs ORDER BY imported_at DESC").fetchall()

    examples: list[dict[str, Any]] = []
    for row in rows:
        sections = resolved_sections(row["content"], row["sections_json"])
        for section in sections:
            if section.get("task_type") == "source_links":
                continue
            examples.append(
                {
                    "task_type": section["task_type"],
                    "prompt": f"请用作者的风格撰写《{row['title']}》中的{section['heading']}部分。",
                    "completion": section["content"],
                    "title": row["title"],
                }
            )
    return examples
