from __future__ import annotations

import hashlib
import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Optional


def now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def current_week_key(today: Optional[date] = None) -> str:
    today = today or date.today()
    year, week, _ = today.isocalendar()
    return f"{year}-W{week:02d}"


def week_key_from_timestamp(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    candidate = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    year, week, _ = parsed.date().isocalendar()
    return f"{year}-W{week:02d}"


def slugify(value: str) -> str:
    normalized = re.sub(r"[^\w\u4e00-\u9fff]+", "-", value.lower()).strip("-")
    return normalized[:80] or "untitled"


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def json_loads(value: Optional[str], default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def sentence_split(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？!?\.])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def paragraph_split(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n\s*\n", text.strip()) if part.strip()]


def compact_whitespace(text: str) -> str:
    normalized_lines: list[str] = []
    blank_pending = False
    for raw_line in text.replace("\r", "\n").split("\n"):
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        if not line:
            blank_pending = True
            continue
        if blank_pending and normalized_lines:
            normalized_lines.append("")
        normalized_lines.append(line)
        blank_pending = False
    cleaned = "\n".join(normalized_lines)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def unique_strings(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            output.append(item)
    return output
