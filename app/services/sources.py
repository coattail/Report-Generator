from __future__ import annotations

import math
import re
import uuid
from collections import Counter, defaultdict
from datetime import date, datetime
from email.utils import parsedate_to_datetime
from typing import Any, Optional
from urllib.parse import urljoin

import feedparser
import httpx
from bs4 import BeautifulSoup

from ..config import DEFAULT_SOURCE_SEEDS, TOPIC_REFINEMENT_LIMIT
from ..db import get_connection
from ..utils import compact_whitespace, current_week_key, hash_text, json_dumps, json_loads, now_iso, slugify, unique_strings, week_key_from_timestamp
from .corpus import latest_style_profile, resolved_sections
from .editorial_ai import OpenAIEditorialClient


CATEGORY_KEYWORDS = {
    "us_economy": ["fed", "inflation", "jobs", "treasury", "consumer", "economy", "cpi", "ppi", "rate"],
    "us_politics": ["white house", "senate", "president", "election", "sec", "congress", "tariff", "policy"],
    "china_macro": ["中国", "财政", "央行", "地产", "消费", "出口", "人民币", "工业"],
    "china_politics": ["国务院", "政治局", "外交部", "政府工作报告", "要闻", "政策"],
    "us_technology": ["ai", "chip", "tech", "semiconductor", "cloud", "apple", "nvidia", "meta", "microsoft"],
    "cross_strait": ["台海", "台湾", "赖清德", "民进党", "国民党", "解放军", "台军", "金门", "马祖", "taiwan", "taiwan strait"],
    "russia_ukraine": ["俄乌", "乌克兰", "俄罗斯", "普京", "泽连斯基", "停火", "北约", "ukraine", "russia", "putin", "zelensky", "nato"],
    "middle_east": ["伊朗", "以色列", "哈马斯", "胡塞", "叙利亚", "中东", "德黑兰", "gaza", "iran", "israel", "hamas", "houthi", "middle east"],
    "global_geopolitics": ["中美", "美俄", "关税", "制裁", "外交", "联盟", "地缘政治", "贸易战", "联合国", "安全会议", "us-china", "tariff", "sanction", "diplomacy", "geopolitics"],
}

CATEGORY_PRIORITY = {
    "cross_strait": 5,
    "russia_ukraine": 5,
    "middle_east": 5,
    "global_geopolitics": 4,
    "us_politics": 3,
    "china_politics": 3,
    "us_economy": 2,
    "china_macro": 2,
    "us_technology": 2,
}

CATEGORY_LABELS = {
    "us_economy": "美国经济",
    "us_politics": "美国政治",
    "china_macro": "中国宏观",
    "china_politics": "中国政治",
    "us_technology": "美国科技",
    "cross_strait": "台海",
    "russia_ukraine": "俄乌",
    "middle_east": "中东",
    "global_geopolitics": "国际政治",
}

WEAK_TOPIC_PATTERNS = [
    r"\bwhere to buy\b",
    r"\bhow to\b",
    r"\bpreorder\b",
    r"\bbuy\b",
    r"\bprice\b",
    r"\bpricing\b",
    r"\brelease date\b",
    r"\bdiscount\b",
    r"\bshopping\b",
    r"\bguide\b",
    r"\bhands-on\b",
    r"\bfirst look\b",
    r"\baccessories\b",
    r"哪里买",
    r"怎么买",
    r"购买",
    r"预购",
    r"开箱",
    r"上手",
    r"体验",
    r"价格",
    r"折扣",
]

TOPIC_RECOMMENDATION_STOPWORDS = {
    "about",
    "after",
    "and",
    "before",
    "board",
    "bbc",
    "briefing",
    "china",
    "company",
    "content",
    "federal",
    "google",
    "government",
    "group",
    "html",
    "https",
    "market",
    "press",
    "report",
    "says",
    "service",
    "startup",
    "system",
    "techcrunch",
    "this",
    "its",
    "live",
    "article",
    "verge",
    "week",
    "www",
    "with",
    "world",
    "news",
    "中国",
    "全国",
    "会议",
    "工作",
    "政府",
    "发展",
    "经济",
    "社会",
    "系统",
    "市场",
    "数据",
    "显示",
    "链接",
    "评述",
    "相关链接",
    "市场评述",
}

CORPUS_TOPIC_HEADING_PATTERNS = (
    re.compile(r"^[（(]\d+[）)]"),
    re.compile(r"^(topic|section)\b", re.I),
)
CORPUS_TOPIC_HEADING_EXCLUSIONS = {
    "相关链接",
    "评述",
    "市场评述",
    "结尾",
    "总结",
    "价值投资视角总结",
    "正文",
}

_GENERIC_TITLE_PATTERNS = [
    re.compile(r"\s*[-|_]\s*(the white house|bbc news 中文|bbc news|voa 中文|voa|techcrunch|the verge|中国政府网).*$", re.I),
    re.compile(r"^(briefing room|latest news|news|首页)$", re.I),
]
_GENERIC_SENTENCE_PATTERNS = [
    re.compile(r"copyright", re.I),
    re.compile(r"all rights reserved", re.I),
    re.compile(r"subscribe", re.I),
    re.compile(r"sign up", re.I),
    re.compile(r"cookies", re.I),
    re.compile(r"privacy policy", re.I),
    re.compile(r"阅读更多"),
    re.compile(r"点击.*查看"),
]
_OFFICIAL_SOURCE_IDS = {"fed-press", "sec-press", "white-house", "gov-cn"}
_TOPIC_REFINEMENT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "title": {"type": "string"},
        "event_summary": {"type": "string"},
        "key_facts": {"type": "array", "items": {"type": "string"}},
        "entities": {"type": "array", "items": {"type": "string"}},
        "impact_paths": {"type": "array", "items": {"type": "string"}},
        "allowed_conclusions": {"type": "array", "items": {"type": "string"}},
        "forbidden_claims": {"type": "array", "items": {"type": "string"}},
        "confidence_note": {"type": "string"},
        "editorial_flags": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "title",
        "event_summary",
        "key_facts",
        "entities",
        "impact_paths",
        "allowed_conclusions",
        "forbidden_claims",
        "confidence_note",
        "editorial_flags",
    ],
}


def seed_sources() -> None:
    with get_connection() as connection:
        existing = {row["id"] for row in connection.execute("SELECT id FROM sources").fetchall()}
        for source in DEFAULT_SOURCE_SEEDS:
            if source["id"] in existing:
                continue
            connection.execute(
                """
                INSERT INTO sources (
                  id, name, base_url, feed_url, page_url, language, source_type,
                  categories_json, enabled, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                """,
                (
                    source["id"],
                    source["name"],
                    source.get("base_url"),
                    source.get("feed_url", ""),
                    source.get("page_url", ""),
                    source.get("language", "en"),
                    source.get("source_type", "media"),
                    json_dumps(source.get("categories", [])),
                    now_iso(),
                    now_iso(),
                ),
            )


def list_sources() -> list[dict[str, Any]]:
    with get_connection() as connection:
        rows = connection.execute("SELECT * FROM sources ORDER BY name").fetchall()
    return [
        {
            "id": row["id"],
            "name": row["name"],
            "base_url": row["base_url"],
            "feed_url": row["feed_url"],
            "page_url": row["page_url"],
            "language": row["language"],
            "source_type": row["source_type"],
            "categories": json_loads(row["categories_json"], []),
            "enabled": bool(row["enabled"]),
        }
        for row in rows
    ]


def _parse_published(entry: dict[str, Any]) -> str:
    for key in ["published", "updated"]:
        raw = entry.get(key)
        if raw:
            try:
                return parsedate_to_datetime(raw).isoformat()
            except (TypeError, ValueError):
                continue
    return now_iso()


def _clean_title_candidate(value: str) -> str:
    title = compact_whitespace(value or "")
    for pattern in _GENERIC_TITLE_PATTERNS:
        title = pattern.sub("", title).strip(" -|_")
    return title.strip()


def _best_page_title(soup: BeautifulSoup, fallback_url: str) -> str:
    meta_candidates = [
        tag.get("content", "")
        for tag in soup.find_all("meta")
        if tag.get("property") in {"og:title"} or tag.get("name") in {"twitter:title"}
    ]
    tag_candidates = [element.get_text(" ", strip=True) for element in soup.find_all(["h1", "h2"], limit=3)]
    candidates = meta_candidates + tag_candidates + ([soup.title.text] if soup.title and soup.title.text else [])
    for candidate in candidates:
        cleaned = _clean_title_candidate(candidate)
        if len(cleaned) >= 10 and not any(pattern.match(cleaned) for pattern in _GENERIC_TITLE_PATTERNS):
            return cleaned
    return fallback_url


def _split_sentences(*parts: str) -> list[str]:
    sentences: list[str] = []
    for part in parts:
        text = compact_whitespace(part or "")
        if not text:
            continue
        for sentence in re.split(r"(?<=[。！？.!?])\s+", text):
            cleaned = compact_whitespace(sentence).strip("；; ")
            if len(cleaned) < 18:
                continue
            if any(pattern.search(cleaned) for pattern in _GENERIC_SENTENCE_PATTERNS):
                continue
            if cleaned not in sentences:
                sentences.append(cleaned)
    return sentences


def _sentence_score(sentence: str, category: str, entities: list[str]) -> float:
    lowered = sentence.lower()
    score = 0.0
    score += min(2.0, sum(0.5 for entity in entities[:8] if entity and entity in sentence))
    score += min(2.0, sum(0.35 for keyword in CATEGORY_KEYWORDS.get(category, []) if keyword.lower() in lowered))
    if any(char.isdigit() for char in sentence):
        score += 0.5
    if "表示" in sentence or "announce" in lowered or "said" in lowered:
        score += 0.3
    if len(sentence) > 120:
        score -= 0.3
    return score


def _select_key_facts(bucket: list[dict[str, Any]], category: str, entities: list[str], limit: int = 6) -> list[str]:
    scored: list[tuple[float, str]] = []
    for article in bucket:
        for sentence in _split_sentences(article.get("summary", ""), article.get("content", "")[:900]):
            score = _sentence_score(sentence, category, entities)
            scored.append((score, sentence))
    scored.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
    facts: list[str] = []
    for _, sentence in scored:
        normalized = sentence.rstrip("。.!? ")
        if normalized not in facts:
            facts.append(normalized)
        if len(facts) >= limit:
            break
    if not facts:
        facts = unique_strings(article["title"] for article in bucket[:limit])[:limit]
    return facts[:limit]


def _category_impact_paths(category: str) -> list[str]:
    mapping = {
        "us_politics": ["评估它会不会继续改变美国政策边界、盟友协调和市场对政治风险的定价。", "优先跟踪政策执行而不是单次表态。"],
        "china_politics": ["评估它会不会改变中国对内政策节奏和对外博弈姿态。", "判断重点放在政策连续性与执行力度。"],
        "china_macro": ["评估它会不会改变增长预期、信用扩张和中国资产的定价锚。", "区分短期刺激与中期增长修复。"],
        "us_economy": ["评估它会不会继续改变利率路径、就业与通胀预期。", "核心是宏观数据如何传导到估值框架。"],
        "us_technology": ["评估它会不会改变科技资本开支、盈利兑现和高估值容忍度。", "重点看产业链订单与盈利兑现。"],
        "cross_strait": ["评估它会不会抬升台海风险溢价，并改变中美与区域安全预期。", "判断重点放在可执行动作而非口号。"],
        "russia_ukraine": ["评估它会不会改变欧洲安全、能源链条与援助持续性。", "重点区分战场变化与外交表态。"],
        "middle_east": ["评估它会不会影响中东冲突外溢、能源价格和全球风险偏好。", "重点看冲突是否跨境扩散。"],
        "global_geopolitics": ["评估它会不会继续改变大国博弈、关税制裁和全球供应链预期。", "重点看政策落地与联盟协调。"],
    }
    return mapping.get(category, ["先判断事实强度，再判断它如何传导到政策预期和资产定价。"])


def _derive_cluster_title(bucket: list[dict[str, Any]], category: str, entities: list[str]) -> str:
    if len(bucket) == 1:
        return bucket[0]["title"]
    title_tokens = Counter()
    for article in bucket:
        for token in re.findall(r"[\w\u4e00-\u9fff]{2,}", article["title"].lower()):
            if token in TOPIC_RECOMMENDATION_STOPWORDS:
                continue
            title_tokens[token] += 1
    top_tokens = [token for token, _ in title_tokens.most_common(3)]
    label = CATEGORY_LABELS.get(category, category)
    if entities and top_tokens:
        return f"{entities[0]}与{'、'.join(top_tokens[:2])}成为本周{label}主线"
    if top_tokens:
        return f"{'、'.join(top_tokens[:2])}成为本周{label}焦点"
    return bucket[0]["title"]


def _heuristic_cluster_refinement(
    bucket: list[dict[str, Any]],
    category: str,
    representative_title: str,
    summary: str,
    entities: list[str],
    official_urls: list[str],
    timeline: list[dict[str, Any]],
    editorial_flags: list[str],
) -> tuple[str, str, dict[str, Any]]:
    key_facts = _select_key_facts(bucket, category, entities)
    event_summary = "；".join(key_facts[:2])[:320] or summary or representative_title
    title = _derive_cluster_title(bucket, category, entities)
    impact_paths = _category_impact_paths(category)
    allowed_conclusions = [
        "先交代哪些事实已经确认，再判断它改变了什么政策、产业或风险定价框架。",
        impact_paths[0],
    ]
    if len(official_urls) >= 2:
        confidence_note = "当前主题同时有官方信源和交叉来源支撑，可以给出方向性判断，但不把结论写满。"
    elif official_urls:
        confidence_note = "当前主题有官方信源支撑，但仍需要更多交叉材料验证其持续性。"
    else:
        confidence_note = "当前主题主要依赖媒体与二手材料，结论只能停在趋势与风险提示层。"
    evidence_pack = {
        "event_summary": event_summary,
        "timeline": timeline[:6],
        "key_facts": key_facts,
        "entities": entities[:10],
        "official_sources": official_urls[:4],
        "cross_sources": unique_strings(item["url"] for item in bucket)[:6],
        "impact_paths": impact_paths[:3],
        "confidence_note": confidence_note,
        "editorial_flags": editorial_flags,
        "allowed_conclusions": allowed_conclusions,
        "forbidden_claims": [
            "没有来源支撑的阴谋论、确定性预言或跨主题拼接。",
            "把短期价格波动直接等同于长期价值判断。",
            "把口号式表态直接写成已落地政策。",
        ],
    }
    return title, event_summary, evidence_pack


def _openai_cluster_refinement(
    bucket: list[dict[str, Any]],
    category: str,
    heuristic_title: str,
    heuristic_evidence_pack: dict[str, Any],
) -> Optional[dict[str, Any]]:
    client = OpenAIEditorialClient()
    if not client.available():
        return None
    article_lines = []
    for article in bucket[:5]:
        article_lines.append(
            "\n".join(
                [
                    f"标题：{article['title']}",
                    f"发布时间：{article['published_at']}",
                    f"链接：{article['url']}",
                    f"摘要：{compact_whitespace(article.get('summary', ''))[:220]}",
                    f"正文摘录：{compact_whitespace(article.get('content', ''))[:480]}",
                ]
            )
        )
    input_text = (
        f"主题类别：{CATEGORY_LABELS.get(category, category)}\n"
        f"初始标题：{heuristic_title}\n"
        f"初始摘要：{heuristic_evidence_pack.get('event_summary', '')}\n"
        f"初始关键事实：{'；'.join(heuristic_evidence_pack.get('key_facts', []))}\n"
        f"已有实体：{'、'.join(heuristic_evidence_pack.get('entities', []))}\n\n"
        "候选材料：\n"
        + "\n\n".join(article_lines)
    )
    try:
        refined = client.create_json(
            instructions=(
                "你是中文财经与国际政治订阅作者的事实研究助理。"
                "请把嘈杂候选材料提炼成一个可写周报的主题证据包。"
                "输出必须只保留被材料支持的事实与有限判断，不要写空洞套话，不要引入材料外信息。"
                "标题要像成熟周报里的分析标题，而不是新闻网页标题。"
            ),
            input_text=input_text,
            schema_name="topic_refinement",
            schema=_TOPIC_REFINEMENT_SCHEMA,
            max_output_tokens=900,
        )
    except Exception:
        return None
    return refined

def _extract_article_content(url: str) -> tuple[str, str]:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; LocalBriefingStudio/0.1)"}
    with httpx.Client(timeout=15, follow_redirects=True, headers=headers) as client:
        response = client.get(url)
        response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    title = _best_page_title(soup, url)
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text_blocks = [element.get_text(" ", strip=True) for element in soup.find_all(["p", "article", "main", "h1", "h2"])]
    content = compact_whitespace("\n\n".join(text_blocks))
    return title, content[:12000]


def _sync_rss(source: dict[str, Any]) -> list[dict[str, Any]]:
    feed = feedparser.parse(source["feed_url"])
    articles: list[dict[str, Any]] = []
    for entry in feed.entries[:20]:
        url = entry.get("link")
        if not url:
            continue
        try:
            title, content = _extract_article_content(url)
        except Exception:
            title = entry.get("title", url)
            content = compact_whitespace(BeautifulSoup(entry.get("summary", ""), "html.parser").get_text(" ", strip=True))
        articles.append(
            {
                "title": title or entry.get("title", url),
                "url": url,
                "published_at": _parse_published(entry),
                "summary": compact_whitespace(BeautifulSoup(entry.get("summary", ""), "html.parser").get_text(" ", strip=True))[:600],
                "content": content,
            }
        )
    return articles


def _sync_html_listing(source: dict[str, Any]) -> list[dict[str, Any]]:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; LocalBriefingStudio/0.1)"}
    with httpx.Client(timeout=15, follow_redirects=True, headers=headers) as client:
        response = client.get(source["page_url"])
        response.raise_for_status()
        redirect_match = re.search(r'window\.location\.href\s*=\s*"([^"]+)"', response.text)
        if redirect_match:
            redirected_url = urljoin(str(response.url), redirect_match.group(1))
            response = client.get(redirected_url)
            response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    articles: list[dict[str, Any]] = []
    for link in soup.find_all("a", href=True):
        href = link.get("href", "")
        text = compact_whitespace(link.get_text(" ", strip=True))
        if len(text) < 12:
            continue
        if href.startswith("#"):
            continue
        absolute_url = urljoin(source["base_url"], href)
        if source["base_url"] not in absolute_url:
            continue
        if absolute_url.rstrip("/") == source["page_url"].rstrip("/"):
            continue
        try:
            title, content = _extract_article_content(absolute_url)
        except Exception:
            continue
        articles.append(
            {
                "title": title or text,
                "url": absolute_url,
                "published_at": now_iso(),
                "summary": text[:600],
                "content": content,
            }
        )
        if len(articles) >= 12:
            break
    return articles


def _sync_gov_cn_listing(source: dict[str, Any]) -> list[dict[str, Any]]:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; LocalBriefingStudio/0.1)"}
    list_url = "https://www.gov.cn/yaowen/liebiao/YAOWENLIEBIAO.json"
    with httpx.Client(timeout=20, follow_redirects=True, headers=headers) as client:
        response = client.get(list_url)
        response.raise_for_status()
        payload = response.json()

    articles: list[dict[str, Any]] = []
    for entry in payload[:20]:
        url = entry.get("URL")
        title = compact_whitespace(entry.get("TITLE", ""))
        if not url or not title:
            continue
        try:
            parsed_title, content = _extract_article_content(url)
        except Exception:
            parsed_title = title
            content = title
        articles.append(
            {
                "title": parsed_title or title,
                "url": url,
                "published_at": entry.get("DOCRELPUBTIME") or now_iso(),
                "summary": compact_whitespace(entry.get("SUB_TITLE", "") or title)[:600],
                "content": content,
            }
        )
    return articles


def categorize_article(title: str, summary: str, content: str, fallback_categories: list[str]) -> str:
    title_haystack = (title or "").lower()
    summary_haystack = (summary or "").lower()
    content_haystack = (content or "")[:500].lower()
    score_map = {category: 0.0 for category in CATEGORY_KEYWORDS}
    for category, keywords in CATEGORY_KEYWORDS.items():
        score = 0.0
        for keyword in keywords:
            if keyword in title_haystack:
                score += 2.4
            elif keyword in summary_haystack:
                score += 1.5
            elif keyword in content_haystack:
                score += 0.8
        score_map[category] = score
    best_category, best_score = max(
        score_map.items(),
        key=lambda item: (item[1], CATEGORY_PRIORITY.get(item[0], 0)),
    )
    if best_score == 0 and fallback_categories:
        return fallback_categories[0]
    return best_category


def extract_entities(title: str, content: str) -> list[str]:
    candidates = re.findall(r"\b[A-Z][A-Za-z&\.-]{1,20}\b", title + " " + content[:800])
    chinese = re.findall(r"[\u4e00-\u9fff]{2,8}", title)[:10]
    return unique_strings(candidates[:10] + chinese[:10])[:12]


def _looks_like_weak_topic(*parts: str) -> bool:
    haystack = " ".join(part for part in parts if part).lower()
    return any(re.search(pattern, haystack) for pattern in WEAK_TOPIC_PATTERNS)


def _tokenize_preference_text(text: str, limit: int = 36) -> list[str]:
    normalized = (text or "").lower()
    normalized = re.sub(r"https?://\S+", " ", normalized)
    normalized = re.sub(r"www\.\S+", " ", normalized)
    english = re.findall(r"[a-z][a-z0-9\-\+]{2,}", normalized)
    chinese = re.findall(r"[\u4e00-\u9fff]{2,8}", text or "")
    tokens: list[str] = []
    for token in english + chinese:
        if token in TOPIC_RECOMMENDATION_STOPWORDS:
            continue
        if token.isdigit():
            continue
        if token not in tokens:
            tokens.append(token)
        if len(tokens) >= limit:
            break
    return tokens


def _rank_priority_keywords(counter: Counter[str], limit: int = 12) -> list[str]:
    category_keywords = {
        keyword
        for keywords in CATEGORY_KEYWORDS.values()
        for keyword in keywords
        if len(keyword) >= 2 and keyword not in TOPIC_RECOMMENDATION_STOPWORDS
    }
    ranked = sorted(
        counter.items(),
        key=lambda item: (
            item[0] in category_keywords,
            item[1],
            len(item[0]) <= 4,
            -len(item[0]),
            item[0],
        ),
        reverse=True,
    )
    keywords: list[str] = []
    for token, _ in ranked:
        if token.startswith("http") or "." in token or "/" in token:
            continue
        keywords.append(token)
        if len(keywords) >= limit:
            break
    return keywords


def _infer_category_from_text(text: str) -> str | None:
    haystack = (text or "").lower()
    scored = []
    for category, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in haystack)
        if score:
            scored.append((score, CATEGORY_PRIORITY.get(category, 0), category))
    if not scored:
        return None
    scored.sort(reverse=True)
    return scored[0][2]


def _is_corpus_topic_section(heading: str, task_type: str) -> bool:
    normalized_heading = heading.strip().strip("：: ")
    if heading.startswith("每周订阅内容") and task_type == "intro":
        return False
    if not normalized_heading:
        return task_type == "topic_section"
    if normalized_heading in CORPUS_TOPIC_HEADING_EXCLUSIONS:
        return False
    if normalized_heading.startswith("相关链接"):
        return False
    return task_type == "topic_section" or any(pattern.match(heading) for pattern in CORPUS_TOPIC_HEADING_PATTERNS)


def _build_corpus_topic_signals(limit_docs: int = 110) -> dict[str, Any]:
    positive_terms: Counter[str] = Counter()
    category_bias: Counter[str] = Counter()
    category_examples: dict[str, list[str]] = defaultdict(list)
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT title, content, sections_json
            FROM style_corpus_docs
            ORDER BY imported_at DESC
            LIMIT ?
            """,
            (limit_docs,),
        ).fetchall()

    topic_section_count = 0
    for index, row in enumerate(rows):
        sections = resolved_sections(row["content"], row["sections_json"])
        recency_weight = max(0.55, 1.2 - index * 0.006)
        doc_tokens: set[str] = set()
        doc_categories: set[str] = set()
        for section in sections:
            heading = str(section.get("heading") or "").strip()
            content = str(section.get("content") or "").strip()
            task_type = str(section.get("task_type") or "")
            if not heading and not content:
                continue
            if not _is_corpus_topic_section(heading, task_type):
                continue
            topic_section_count += 1
            sample_text = " ".join(part for part in [heading, content[:180]] if part)
            sample_haystack = sample_text.lower()
            doc_tokens.update(_tokenize_preference_text(sample_text, limit=12))
            category = _infer_category_from_text(sample_text)
            if category:
                doc_categories.add(category)
                if heading and heading not in category_examples[category]:
                    category_examples[category].append(heading)
            for keyword in unique_strings(
                keyword
                for keywords in CATEGORY_KEYWORDS.values()
                for keyword in keywords
                if len(keyword) >= 2 and keyword in sample_haystack and keyword not in TOPIC_RECOMMENDATION_STOPWORDS
            )[:8]:
                doc_tokens.add(keyword)
        for token in doc_tokens:
            positive_terms[token] += round(0.18 * recency_weight, 3)
        for category in doc_categories:
            category_bias[category] += round(0.42 * recency_weight, 3)

    return {
        "document_count": len(rows),
        "topic_section_count": topic_section_count,
        "positive_terms": positive_terms,
        "category_bias": category_bias,
        "category_examples": category_examples,
    }


def topic_preference_overview(limit_docs: int = 110) -> dict[str, Any]:
    signals = _build_corpus_topic_signals(limit_docs)
    category_bias: Counter[str] = signals["category_bias"]
    with get_connection() as connection:
        selection_feedback_count = connection.execute(
            """
            SELECT COUNT(*) AS pair_count
            FROM preference_pairs
            WHERE reason = 'topic_selection_feedback'
            """
        ).fetchone()["pair_count"]
    total = sum(category_bias.values()) or 1.0
    priority_categories = []
    for category, score in category_bias.most_common(6):
        priority_categories.append(
            {
                "category": category,
                "label": CATEGORY_LABELS.get(category, category),
                "score": round(score, 2),
                "share": round(score / total * 100, 1),
                "examples": signals["category_examples"].get(category, [])[:3],
            }
        )
    priority_keywords = _rank_priority_keywords(signals["positive_terms"])
    return {
        "document_count": signals["document_count"],
        "topic_section_count": signals["topic_section_count"],
        "selection_feedback_count": int(selection_feedback_count or 0),
        "priority_categories": priority_categories,
        "priority_keywords": priority_keywords,
    }


def _build_recommendation_profile() -> dict[str, Any]:
    positive_terms: Counter[str] = Counter()
    negative_terms: Counter[str] = Counter()
    category_bias: Counter[str] = Counter()
    style_profile = latest_style_profile()
    corpus_signals = _build_corpus_topic_signals()

    for text in (style_profile.get("worldview_notes") or [])[:6] + (style_profile.get("preferred_patterns") or [])[:4]:
        for token in _tokenize_preference_text(text, limit=20):
            positive_terms[token] += 0.8
        category = _infer_category_from_text(text)
        if category:
            category_bias[category] += 0.4

    for token, weight in corpus_signals["positive_terms"].items():
        positive_terms[token] += weight
    for category, weight in corpus_signals["category_bias"].items():
        category_bias[category] += weight

    with get_connection() as connection:
        scored_sections = connection.execute(
            """
            SELECT
              sec.title,
              sec.content,
              sc.publishability,
              COALESCE(sc.notes, '') AS notes,
              COALESCE(sc.flags_json, '[]') AS flags_json
            FROM issue_sections sec
            JOIN (
              SELECT section_id, MAX(created_at) AS latest_created_at
              FROM section_scores
              WHERE section_id IS NOT NULL
              GROUP BY section_id
            ) latest ON latest.section_id = sec.id
            JOIN section_scores sc
              ON sc.section_id = latest.section_id
             AND sc.created_at = latest.latest_created_at
            ORDER BY latest.latest_created_at DESC
            LIMIT 120
            """
        ).fetchall()
        preference_rows = connection.execute(
            """
            SELECT chosen_text, rejected_text, COALESCE(reason, '') AS reason
            FROM preference_pairs
            ORDER BY created_at DESC
            LIMIT 120
            """
        ).fetchall()

    for row in scored_sections:
        basis_text = f"{row['title']} {row['content'][:320]} {row['notes']}"
        publishability = int(row["publishability"] or 3)
        flags = json_loads(row["flags_json"], [])
        tokens = _tokenize_preference_text(basis_text)
        category = _infer_category_from_text(basis_text)
        if publishability >= 4:
            for token in tokens:
                positive_terms[token] += 1.2
            if category:
                category_bias[category] += 0.8
        elif publishability <= 2 or any("选题" in flag or "偏消费" in flag for flag in flags):
            for token in tokens:
                negative_terms[token] += 1.4
            if category:
                category_bias[category] -= 0.6

    for row in preference_rows:
        chosen_tokens = _tokenize_preference_text(row["chosen_text"], limit=30)
        rejected_tokens = _tokenize_preference_text(row["rejected_text"], limit=30)
        pair_weight = 1.4 if row["reason"] == "topic_selection_feedback" else 1.0
        for token in chosen_tokens:
            positive_terms[token] += pair_weight
        for token in rejected_tokens:
            negative_terms[token] += pair_weight
        chosen_category = _infer_category_from_text(row["chosen_text"])
        rejected_category = _infer_category_from_text(row["rejected_text"])
        if chosen_category:
            category_bias[chosen_category] += 0.8 if row["reason"] == "topic_selection_feedback" else 0.6
        if rejected_category:
            category_bias[rejected_category] -= 0.55 if row["reason"] == "topic_selection_feedback" else 0.4

    return {
        "positive_terms": positive_terms,
        "negative_terms": negative_terms,
        "category_bias": category_bias,
    }


def recommend_topics(week_key: Optional[str] = None, min_count: int = 10, max_count: int = 15) -> list[dict[str, Any]]:
    topics = get_topics(week_key)
    if not topics:
        return []

    profile = _build_recommendation_profile()
    positive_terms: Counter[str] = profile["positive_terms"]
    negative_terms: Counter[str] = profile["negative_terms"]
    category_bias: Counter[str] = profile["category_bias"]
    enriched: list[dict[str, Any]] = []

    for topic in topics:
        topic_text = " ".join(
            [
                topic["title"],
                topic["summary"],
                " ".join(topic.get("entities", [])),
                topic.get("evidence_pack", {}).get("event_summary", ""),
            ]
        )
        tokens = _tokenize_preference_text(topic_text)
        positive_hits = [(token, positive_terms[token]) for token in tokens if positive_terms[token] > 0]
        negative_hits = [(token, negative_terms[token]) for token in tokens if negative_terms[token] > 0]
        positive_hits.sort(key=lambda item: item[1], reverse=True)
        negative_hits.sort(key=lambda item: item[1], reverse=True)
        positive_bonus = round(min(2.6, sum(weight for _, weight in positive_hits[:4]) * 0.35), 2)
        negative_penalty = round(min(2.4, sum(weight for _, weight in negative_hits[:3]) * 0.45), 2)
        official_bonus = round(min(1.6, len(topic["evidence_pack"].get("official_sources", [])) * 0.45), 2)
        category_bonus = round(max(-1.8, min(2.2, category_bias.get(topic["category"], 0.0))), 2)
        weak_penalty = 3.5 if "weak_topic" in topic["evidence_pack"].get("editorial_flags", []) else 0.0
        recommendation_score = round(
            float(topic["score"]) + positive_bonus + official_bonus + category_bonus - negative_penalty - weak_penalty,
            2,
        )
        reasons = []
        if positive_hits:
            reasons.append("匹配已学习偏好：" + " / ".join(token for token, _ in positive_hits[:3]))
        if category_bonus > 0.2:
            reasons.append(f"近期更偏好 {topic['category']} 方向")
        if official_bonus > 0:
            reasons.append(f"官方信源 {len(topic['evidence_pack'].get('official_sources', []))} 个")
        if weak_penalty:
            reasons.append("疑似消费/弱选题，已自动降权")
        if not reasons:
            reasons.append("按系统基础重要性排序进入候选")
        enriched.append(
            {
                **topic,
                "recommendation_score": recommendation_score,
                "recommendation_reasons": reasons,
                "recommended": False,
            }
        )

    enriched.sort(
        key=lambda item: (
            item["recommendation_score"],
            len(item["evidence_pack"].get("official_sources", [])),
            item["score"],
            item["title"],
        ),
        reverse=True,
    )
    if len(enriched) < min_count:
        target_count = len(enriched)
    else:
        target_count = min(max_count, max(min_count, min(len(enriched), 12)))

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for topic in enriched:
        buckets[topic["category"]].append(topic)
    category_order = sorted(
        buckets,
        key=lambda category: buckets[category][0]["recommendation_score"],
        reverse=True,
    )
    max_per_category = max(2, (target_count + max(len(category_order), 1) - 1) // max(len(category_order), 1) + 1)
    selected: list[dict[str, Any]] = []
    category_counts: Counter[str] = Counter()
    selected_ids: set[str] = set()

    while len(selected) < target_count:
        progressed = False
        for category in category_order:
            bucket = buckets[category]
            while bucket and bucket[0]["id"] in selected_ids:
                bucket.pop(0)
            if not bucket:
                continue
            if category_counts[category] >= max_per_category:
                continue
            topic = bucket.pop(0)
            selected.append({**topic, "recommended": True})
            selected_ids.add(topic["id"])
            category_counts[category] += 1
            progressed = True
            if len(selected) >= target_count:
                break
        if not progressed:
            break

    if len(selected) < target_count:
        for topic in enriched:
            if topic["id"] in selected_ids:
                continue
            selected.append({**topic, "recommended": True})
            selected_ids.add(topic["id"])
            if len(selected) >= target_count:
                break

    selected.sort(key=lambda item: item["recommendation_score"], reverse=True)
    return selected


def store_articles(source: dict[str, Any], articles: list[dict[str, Any]]) -> int:
    created = 0
    fallback_categories = source["categories"]
    with get_connection() as connection:
        for article in articles:
            content = compact_whitespace(article["content"])
            if len(content) < 120:
                continue
            category = categorize_article(article["title"], article["summary"], content, fallback_categories)
            entities = extract_entities(article["title"], content)
            content_hash = hash_text(content)
            try:
                connection.execute(
                    """
                    INSERT INTO raw_articles (
                      id, source_id, title, url, published_at, language, category, author, summary,
                      content, content_hash, tags_json, entities_json, trust_level, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        uuid.uuid4().hex,
                        source["id"],
                        article["title"],
                        article["url"],
                        article["published_at"],
                        source["language"],
                        category,
                        "",
                        article["summary"],
                        content,
                        content_hash,
                        json_dumps(source["categories"]),
                        json_dumps(entities),
                        "high" if source["source_type"] == "official" else "medium",
                        now_iso(),
                    ),
                )
                created += 1
            except Exception:
                continue
    return created


def _article_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_tokens = {token for token in re.findall(r"[\w\u4e00-\u9fff]{2,}", left["title"].lower())}
    right_tokens = {token for token in re.findall(r"[\w\u4e00-\u9fff]{2,}", right["title"].lower())}
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
    entity_overlap = len(set(left["entities"]) & set(right["entities"])) / max(len(set(left["entities"]) | set(right["entities"])), 1)
    category_bonus = 0.15 if left["category"] == right["category"] else 0
    return overlap * 0.7 + entity_overlap * 0.15 + category_bonus


def rebuild_topic_clusters(week_key: Optional[str] = None) -> list[dict[str, Any]]:
    week_key = week_key or current_week_key()
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
              art.id,
              art.title,
              art.url,
              art.published_at,
              art.category,
              art.summary,
              art.content,
              art.entities_json,
              art.source_id,
              COALESCE(src.categories_json, '[]') AS source_categories_json
            FROM raw_articles art
            LEFT JOIN sources src ON src.id = art.source_id
            ORDER BY art.published_at DESC, art.created_at DESC
            """
        ).fetchall()

    articles = [
        {
            "id": row["id"],
            "title": row["title"],
            "url": row["url"],
            "published_at": row["published_at"],
            "category": categorize_article(
                row["title"],
                row["summary"],
                row["content"],
                json_loads(row["source_categories_json"], []),
            ),
            "summary": row["summary"],
            "content": row["content"],
            "entities": json_loads(row["entities_json"], []),
            "source_id": row["source_id"],
        }
        for row in rows
        if week_key_from_timestamp(row["published_at"]) == week_key
    ]

    buckets: list[list[dict[str, Any]]] = []
    for article in articles:
        placed = False
        for bucket in buckets:
            if _article_similarity(article, bucket[0]) >= 0.28:
                bucket.append(article)
                placed = True
                break
        if not placed:
            buckets.append([article])

    clusters: list[dict[str, Any]] = []
    for bucket in buckets:
        titles = Counter(item["title"] for item in bucket)
        category = Counter(item["category"] for item in bucket).most_common(1)[0][0]
        entities = unique_strings(entity for item in bucket for entity in item["entities"])
        representative_title = titles.most_common(1)[0][0]
        timeline = sorted(
            [
                {
                    "published_at": item["published_at"],
                    "title": item["title"],
                    "url": item["url"],
                }
                for item in bucket
            ],
            key=lambda item: item["published_at"] or "",
        )
        summary = "；".join(item["summary"] for item in bucket if item["summary"])[:500]
        official_urls = [item["url"] for item in bucket if item["source_id"] in _OFFICIAL_SOURCE_IDS]
        editorial_flags: list[str] = []
        if any(_looks_like_weak_topic(item["title"], item["summary"], item["content"][:400]) for item in bucket):
            editorial_flags.append("weak_topic")
        refined_title, refined_summary, evidence_pack = _heuristic_cluster_refinement(
            bucket,
            category,
            representative_title,
            summary,
            entities,
            official_urls,
            timeline,
            editorial_flags,
        )
        base_score = round(min(10.0, math.log(len(bucket) + 1, 2) * 3 + len(official_urls) * 1.2), 2)
        penalty = 4.0 if "weak_topic" in editorial_flags else 0.0
        clusters.append(
            {
                "id": uuid.uuid4().hex,
                "week_key": week_key,
                "title": refined_title,
                "slug": slugify(refined_title),
                "summary": refined_summary or summary or bucket[0]["content"][:280],
                "category": category,
                "score": max(0.1, round(base_score - penalty, 2)),
                "article_ids": [item["id"] for item in bucket],
                "entities": entities[:12],
                "timeline": timeline[:8],
                "evidence_pack": evidence_pack,
                "latest_published_at": max((item["published_at"] or "" for item in bucket), default=""),
                "_bucket": bucket,
            }
        )

    clusters.sort(key=lambda item: (item["score"], item["latest_published_at"], item["title"]), reverse=True)
    for cluster in clusters[:TOPIC_REFINEMENT_LIMIT]:
        refined = _openai_cluster_refinement(
            cluster["_bucket"],
            cluster["category"],
            cluster["title"],
            cluster["evidence_pack"],
        )
        if not refined:
            continue
        cluster["title"] = compact_whitespace(refined.get("title") or cluster["title"])[:120] or cluster["title"]
        cluster["slug"] = slugify(cluster["title"])
        cluster["summary"] = compact_whitespace(refined.get("event_summary") or cluster["summary"])[:500] or cluster["summary"]
        cluster["evidence_pack"].update(
            {
                "event_summary": compact_whitespace(refined.get("event_summary") or cluster["evidence_pack"].get("event_summary"))[:500],
                "key_facts": unique_strings(compact_whitespace(item) for item in refined.get("key_facts", []))[:8] or cluster["evidence_pack"].get("key_facts", []),
                "entities": unique_strings(compact_whitespace(item) for item in refined.get("entities", []))[:10] or cluster["evidence_pack"].get("entities", []),
                "impact_paths": unique_strings(compact_whitespace(item) for item in refined.get("impact_paths", []))[:4] or cluster["evidence_pack"].get("impact_paths", []),
                "allowed_conclusions": unique_strings(compact_whitespace(item) for item in refined.get("allowed_conclusions", []))[:4] or cluster["evidence_pack"].get("allowed_conclusions", []),
                "forbidden_claims": unique_strings(compact_whitespace(item) for item in refined.get("forbidden_claims", []))[:4] or cluster["evidence_pack"].get("forbidden_claims", []),
                "confidence_note": compact_whitespace(refined.get("confidence_note") or cluster["evidence_pack"].get("confidence_note")),
                "editorial_flags": unique_strings(list(cluster["evidence_pack"].get("editorial_flags", [])) + list(refined.get("editorial_flags", []))),
            }
        )
    strong_clusters = [cluster for cluster in clusters if "weak_topic" not in cluster["evidence_pack"].get("editorial_flags", [])]
    weak_clusters = [cluster for cluster in clusters if "weak_topic" in cluster["evidence_pack"].get("editorial_flags", [])]
    if len(strong_clusters) >= 5:
        clusters = strong_clusters[:18]
    else:
        clusters = (strong_clusters + weak_clusters)[:18]
    for cluster in clusters:
        cluster.pop("_bucket", None)

    with get_connection() as connection:
        connection.execute("DELETE FROM topic_clusters WHERE week_key = ?", (week_key,))
        for cluster in clusters:
            connection.execute(
                """
                INSERT INTO topic_clusters (
                  id, week_key, title, slug, summary, category, score, article_ids_json,
                  entities_json, timeline_json, evidence_pack_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cluster["id"],
                    cluster["week_key"],
                    cluster["title"],
                    cluster["slug"],
                    cluster["summary"],
                    cluster["category"],
                    cluster["score"],
                    json_dumps(cluster["article_ids"]),
                    json_dumps(cluster["entities"]),
                    json_dumps(cluster["timeline"]),
                    json_dumps(cluster["evidence_pack"]),
                    now_iso(),
                    now_iso(),
                ),
            )
    return clusters


def sync_sources(week_key: Optional[str] = None) -> dict[str, Any]:
    week_key = week_key or current_week_key()
    total_created = 0
    sources = [source for source in list_sources() if source["enabled"]]
    for source in sources:
        try:
            if source["id"] == "gov-cn":
                articles = _sync_gov_cn_listing(source)
            elif source["feed_url"]:
                articles = _sync_rss(source)
            elif source["page_url"]:
                articles = _sync_html_listing(source)
            else:
                articles = []
        except Exception:
            articles = []
        total_created += store_articles(source, articles)
    clusters = rebuild_topic_clusters(week_key)
    return {
        "synced_articles": total_created,
        "cluster_count": len(clusters),
        "week_key": week_key,
    }


def get_topics(week_key: Optional[str] = None) -> list[dict[str, Any]]:
    week_key = week_key or current_week_key()
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT * FROM topic_clusters WHERE week_key = ? ORDER BY score DESC, updated_at DESC
            """,
            (week_key,),
        ).fetchall()
    return [
        {
            "id": row["id"],
            "week_key": row["week_key"],
            "title": row["title"],
            "summary": row["summary"],
            "category": row["category"],
            "score": row["score"],
            "article_ids": json_loads(row["article_ids_json"], []),
            "entities": json_loads(row["entities_json"], []),
            "timeline": json_loads(row["timeline_json"], []),
            "evidence_pack": json_loads(row["evidence_pack_json"], {}),
        }
        for row in rows
    ]


def topics_overview(week_key: Optional[str] = None) -> dict[str, Any]:
    topics = get_topics(week_key)
    recommended_topics = recommend_topics(week_key)
    recommended_ids = {topic["id"] for topic in recommended_topics}
    grouped = defaultdict(list)
    for topic in topics:
        grouped[topic["category"]].append(topic)
    return {
        "week_key": week_key or current_week_key(),
        "count": len(topics),
        "recommended_count": len(recommended_topics),
        "recommended_topics": recommended_topics,
        "other_topics": [topic for topic in topics if topic["id"] not in recommended_ids],
        "by_category": dict(grouped),
        "topics": topics,
    }
