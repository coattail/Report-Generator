from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import uuid
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import DB_PATH
from app.db import get_connection, init_db
from app.services.corpus import build_style_profile, derive_structure_tags, split_sections
from app.services.training import run_preference_training, run_sft_training
from app.utils import hash_text, json_dumps, now_iso, paragraph_split, sentence_split


WEEKLY_TITLE_PATTERN = re.compile(r"^每周订阅内容\s*\d+")
SECTION_PATTERN = re.compile(r"《(?P<title>.+?)》中的(?P<section>.+?)部分")
NUMBERED_PATTERN = re.compile(r"^[（(](?P<index>\d+)[）)]\s*(?P<label>.+)$")
STANCE_MARKERS = ("我认为", "我觉得", "我的判断", "这意味着", "更重要的是", "关键是")
MARKET_MARKERS = ("估值", "利率", "通胀", "风险偏好", "科技", "美股", "纳指", "标普", "美债", "市场")


@dataclass
class TopicBucket:
    heading: str
    index: int | None = None
    links: list[str] = field(default_factory=list)
    bodies: list[str] = field(default_factory=list)


@dataclass
class WeeklyDraft:
    title: str
    intro_candidates: list[str] = field(default_factory=list)
    explicit_topics: "OrderedDict[str, TopicBucket]" = field(default_factory=OrderedDict)
    loose_topics: list[str] = field(default_factory=list)
    market_candidates: list[tuple[str | None, str]] = field(default_factory=list)
    conclusion_candidates: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover the style corpus from prior MLX SFT datasets.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=ROOT_DIR / "data" / "training" / "mlx_sft" / "e181b9b09c93495a8b71bdbdfe06dd11",
        help="Folder containing prior train/valid/test JSONL files.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Print the recovery summary without writing to the database.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Recover the corpus and rebuild the style profile, but skip simulated SFT/preference retraining.",
    )
    return parser.parse_args()


def _normalize_title(title: str) -> str:
    normalized = re.sub(r"\s+", " ", str(title or "")).strip()
    match = re.match(r"^(每周订阅内容)\s*(\d+)$", normalized)
    if match:
        return f"{match.group(1)} {match.group(2)}"
    return normalized


def _extract_requested_section(prompt: str) -> str:
    match = SECTION_PATTERN.search(prompt or "")
    if not match:
        return ""
    section = re.sub(r"\s+", " ", match.group("section")).strip()
    return section.replace("：", ":")


def _split_completion(text: str) -> tuple[list[str], str]:
    parsed = split_sections((text or "").strip())
    links = [section["content"].strip() for section in parsed if section.get("task_type") == "source_links"]
    bodies = [section["content"].strip() for section in parsed if section.get("task_type") != "source_links"]
    return links, "\n\n".join(part for part in bodies if part).strip()


def _is_link_only(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return True
    if stripped.startswith("相关链接"):
        payload = re.sub(r"^相关链接\d*\s*[:：]?\s*", "", stripped)
        return payload.startswith(("http://", "https://")) or "\n" not in payload
    return False


def _is_market_heading(heading: str) -> bool:
    return "市场评述" in heading


def _is_conclusion_heading(heading: str) -> bool:
    return heading in {"价值投资视角总结", "本周总判断"}


def _is_intro_heading(heading: str) -> bool:
    return heading in {"导语", "开篇", "前言"}


def _clean_sentence_label(text: str, limit: int = 24) -> str:
    sentence = sentence_split((text or "").strip()[:160])
    label = sentence[0] if sentence else (text or "").strip()
    label = re.sub(r"https?://\S+", "", label)
    label = re.sub(r"^[（(]?\d+[）)]?\s*", "", label)
    label = re.sub(r"\s+", "", label)
    label = label.strip("：:，。；;,. ")
    if not label:
        label = "重点议题"
    if len(label) > limit:
        label = label[:limit].rstrip("，。；;,. ") + "..."
    return label


def _text_score(text: str) -> int:
    content = (text or "").strip()
    if not content:
        return -10_000
    score = len(content)
    score += sum(100 for marker in STANCE_MARKERS if marker in content)
    score += sum(20 for marker in MARKET_MARKERS if marker in content)
    if _is_link_only(content):
        score -= 5_000
    return score


def _best_text(candidates: Iterable[str]) -> str:
    deduped: OrderedDict[str, None] = OrderedDict()
    for candidate in candidates:
        cleaned = (candidate or "").strip()
        if cleaned:
            deduped.setdefault(cleaned, None)
    if not deduped:
        return ""
    return max(deduped.keys(), key=_text_score)


def _ordered_links(candidates: Iterable[str]) -> list[str]:
    ordered: OrderedDict[str, None] = OrderedDict()
    for candidate in candidates:
        cleaned = (candidate or "").strip()
        if cleaned:
            ordered.setdefault(cleaned, None)
    return list(ordered.keys())


def _topic_bucket(record: WeeklyDraft, heading: str) -> TopicBucket:
    bucket = record.explicit_topics.get(heading)
    if bucket:
        return bucket
    match = NUMBERED_PATTERN.match(heading)
    bucket = TopicBucket(heading=heading, index=int(match.group("index")) if match else None)
    record.explicit_topics[heading] = bucket
    return bucket


def _append_row(record: WeeklyDraft, section_heading: str, task_type: str, completion: str) -> None:
    section_heading = (section_heading or "").strip()
    links, body = _split_completion(completion)
    raw = (completion or "").strip()

    if NUMBERED_PATTERN.match(section_heading) and not _is_market_heading(section_heading):
        bucket = _topic_bucket(record, section_heading)
        bucket.links.extend(_ordered_links(links))
        if body and not _is_link_only(body):
            bucket.bodies.append(body)
        elif raw and _is_link_only(raw):
            bucket.links.append(raw)
        return

    if _is_market_heading(section_heading) or task_type == "market_review":
        candidate = body or raw
        if candidate and not _is_link_only(candidate):
            record.market_candidates.append((section_heading or None, candidate))
        return

    if _is_conclusion_heading(section_heading) or task_type == "conclusion":
        candidate = body or raw
        if candidate and not _is_link_only(candidate):
            record.conclusion_candidates.append(candidate)
        return

    if _is_intro_heading(section_heading) or task_type == "intro":
        candidate = body or raw
        if candidate and not _is_link_only(candidate):
            record.intro_candidates.append(candidate)
        return

    if section_heading.startswith("相关链接"):
        if record.explicit_topics:
            last_bucket = next(reversed(record.explicit_topics.values()))
            last_bucket.links.append(raw)
        return

    candidate = body or raw
    if candidate and not _is_link_only(candidate):
        record.loose_topics.append(candidate)


def _build_content(record: WeeklyDraft) -> tuple[str, list[dict[str, str]]]:
    sections: list[dict[str, str]] = []
    content_parts = [record.title]
    intro = _best_text(record.intro_candidates)
    if intro:
        content_parts.extend(["", intro])
        sections.append({"heading": "导语", "content": intro, "task_type": "intro"})

    ordered_topics = sorted(
        record.explicit_topics.values(),
        key=lambda bucket: (bucket.index if bucket.index is not None else 999, bucket.heading),
    )
    loose_topics = list(OrderedDict((topic.strip(), None) for topic in record.loose_topics if topic.strip()).keys())

    for bucket in ordered_topics:
        if not bucket.bodies and loose_topics:
            bucket.bodies.append(loose_topics.pop(0))

    max_index = max((bucket.index or 0 for bucket in ordered_topics), default=0)
    appended_topics: list[TopicBucket] = []
    for topic_text in loose_topics:
        max_index += 1
        appended_topics.append(TopicBucket(heading=f"（{max_index}）{_clean_sentence_label(topic_text)}", index=max_index, bodies=[topic_text]))

    all_topics = ordered_topics + appended_topics
    if not all_topics and intro:
        max_index = 0

    for bucket in all_topics:
        body = _best_text(bucket.bodies)
        links = _ordered_links(bucket.links)
        if not body and not links:
            continue
        content_parts.extend(["", bucket.heading])
        if links:
            link_line = links[0] if links[0].startswith("相关链接") else f"相关链接：{links[0]}"
            content_parts.extend(["", link_line])
            sections.append({"heading": "相关链接", "content": re.sub(r"^相关链接\s*[:：]?\s*", "", link_line).strip(), "task_type": "source_links"})
        if body:
            content_parts.extend(["", "评述：", "", body])
            sections.append({"heading": bucket.heading, "content": body, "task_type": "topic_section"})

    market_text = _best_text(candidate for _, candidate in record.market_candidates)
    if market_text:
        market_heading = next((heading for heading, _ in record.market_candidates if heading and _is_market_heading(heading)), "")
        if not market_heading:
            market_heading = f"（{max_index + 1}）市场评述" if max_index else "市场评述"
        content_parts.extend(["", market_heading, "", market_text])
        sections.append({"heading": market_heading, "content": market_text, "task_type": "market_review"})
        max_index += 1

    conclusion_text = _best_text(record.conclusion_candidates)
    if conclusion_text:
        content_parts.extend(["", "本周总判断", "", conclusion_text])
        sections.append({"heading": "本周总判断", "content": conclusion_text, "task_type": "conclusion"})

    content = "\n".join(part for part in content_parts if part is not None).strip()
    return content, sections


def _load_weekly_drafts(dataset_dir: Path) -> dict[str, WeeklyDraft]:
    records: dict[str, WeeklyDraft] = {}
    for split_name in ("train.jsonl", "valid.jsonl", "test.jsonl"):
        path = dataset_dir / split_name
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                title = _normalize_title(row.get("title", ""))
                if not WEEKLY_TITLE_PATTERN.match(title):
                    continue
                section_heading = _extract_requested_section(row.get("prompt", ""))
                if not section_heading:
                    continue
                record = records.setdefault(title, WeeklyDraft(title=title))
                _append_row(record, section_heading, row.get("task_type", ""), row.get("completion", ""))
    return records


def _preview(records: dict[str, WeeklyDraft]) -> None:
    topic_counts = []
    for record in records.values():
        ordered_topics = sorted(record.explicit_topics.values(), key=lambda bucket: (bucket.index or 999, bucket.heading))
        topic_count = len(ordered_topics) + len(record.loose_topics)
        topic_counts.append(topic_count)
    avg_topics = round(sum(topic_counts) / max(len(topic_counts), 1), 1)
    print(f"Recovered drafts: {len(records)}")
    print(f"Average topic slots per draft: {avg_topics}")
    for title in sorted(records)[:5]:
        content, _ = _build_content(records[title])
        print("----")
        print(title)
        print(content[:900])


def _backup_db() -> Path:
    stamp = now_iso().replace(":", "").replace("-", "").replace("T", "-").split(".")[0]
    backup_path = DB_PATH.with_suffix(f".pre-recovery-{stamp}.bak")
    shutil.copy2(DB_PATH, backup_path)
    return backup_path


def _replace_style_corpus(records: dict[str, WeeklyDraft]) -> dict[str, int]:
    recovered_rows: list[dict[str, str]] = []
    for title, record in sorted(records.items()):
        content, sections = _build_content(record)
        if len(paragraph_split(content)) < 4:
            continue
        recovered_rows.append(
            {
                "id": uuid.uuid4().hex,
                "file_name": f"{title}.recovered.txt",
                "file_type": "recovered_jsonl",
                "imported_at": now_iso(),
                "title": title,
                "content": content,
                "sections_json": json_dumps(sections),
                "metadata_json": json_dumps(
                    {
                        "content_hash": hash_text(content),
                        "paragraph_count": len(paragraph_split(content)),
                        "sentence_count": len(sentence_split(content)),
                        "recovered_from": "mlx_sft_jsonl",
                    }
                ),
                "structure_tags_json": json_dumps(derive_structure_tags(content)),
            }
        )

    with get_connection() as connection:
        connection.execute("DELETE FROM style_corpus_docs")
        connection.execute("DELETE FROM style_profiles")
        for row in recovered_rows:
            connection.execute(
                """
                INSERT INTO style_corpus_docs (
                  id, file_name, file_type, imported_at, title, content,
                  sections_json, metadata_json, structure_tags_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["id"],
                    row["file_name"],
                    row["file_type"],
                    row["imported_at"],
                    row["title"],
                    row["content"],
                    row["sections_json"],
                    row["metadata_json"],
                    row["structure_tags_json"],
                ),
            )

    profile = build_style_profile()
    return {
        "document_count": len(recovered_rows),
        "profile_document_count": int(((profile.get("stats") or {}).get("document_count")) or 0),
    }


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    if not dataset_dir.exists():
        print(f"Dataset directory not found: {dataset_dir}", file=sys.stderr)
        return 1

    init_db()
    records = _load_weekly_drafts(dataset_dir)
    if not records:
        print("No weekly drafts found in the dataset.", file=sys.stderr)
        return 1

    if args.preview:
        _preview(records)
        return 0

    backup_path = _backup_db()
    replace_result = _replace_style_corpus(records)
    print(f"Database backup created at: {backup_path}")
    print(f"Recovered style corpus documents: {replace_result['document_count']}")
    print(f"Style profile document count: {replace_result['profile_document_count']}")

    if not args.skip_training:
        sft_result = run_sft_training()
        preference_result = run_preference_training()
        print(
            "SFT artifact "
            f"{sft_result['artifact_id']} with {sft_result['metrics']['example_count']} examples "
            f"at {sft_result['dataset_path']}"
        )
        print(
            "Preference artifact "
            f"{preference_result['artifact_id']} with {preference_result['metrics']['pair_count']} pairs "
            f"at {preference_result['dataset_path']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
