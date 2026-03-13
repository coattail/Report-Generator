from __future__ import annotations

import re
from typing import Any

_OFF_TOPIC_LEAK_TERMS = [
    "iphone",
    "macbook",
    "preorder",
    "where to buy",
    "导购",
    "开箱",
    "购买指南",
    "房地产",
    "按揭",
    "房贷",
    "lpr",
    "首付",
    "写字楼",
    "商业办公",
    "再贷款",
]
_STANCE_MARKERS = ["我认为", "我觉得", "我的判断", "这意味着", "更重要的是", "关键是"]


def score_candidate(candidate_text: str, evidence_pack: dict[str, Any], style_profile: dict[str, Any]) -> dict[str, Any]:
    facts = evidence_pack.get("key_facts", [])
    coverage = sum(1 for fact in facts[:4] if fact[:12] in candidate_text)
    citations = len(evidence_pack.get("cross_sources", []))
    banned_hits = [phrase for phrase in style_profile.get("banned_phrases", []) if phrase in candidate_text]
    duplicate_penalty = 0.3 if len(re.findall(r"这意味着", candidate_text)) > 1 else 0.0
    stance_bonus = 0.6 if any(marker in candidate_text for marker in _STANCE_MARKERS) else 0.0
    evidence_text = " ".join(
        [
            evidence_pack.get("event_summary", ""),
            " ".join(evidence_pack.get("key_facts", [])),
            " ".join(evidence_pack.get("entities", [])),
        ]
    ).lower()
    leakage_hits = [term for term in _OFF_TOPIC_LEAK_TERMS if term in candidate_text.lower() and term not in evidence_text]
    leakage_penalty = min(4.5, len(leakage_hits) * 1.5)
    score = coverage * 2.5 + min(citations, 4) * 0.5 + stance_bonus - len(banned_hits) * 2 - duplicate_penalty - leakage_penalty
    notes = []
    if coverage < 1:
        notes.append("事实覆盖偏弱。")
    if not evidence_pack.get("official_sources"):
        notes.append("缺少官方信源，后续结论需要保守。")
    if banned_hits:
        notes.append(f"命中禁用表达：{', '.join(banned_hits)}。")
    if duplicate_penalty:
        notes.append("存在重复表达。")
    if not stance_bonus:
        notes.append("作者判断还不够鲜明。")
    if leakage_hits:
        notes.append(f"疑似串文污染：出现当前证据包未支持的内容 {', '.join(leakage_hits)}。")
    return {
        "score": round(score, 2),
        "notes": notes or ["候选内容通过基础审校。"],
    }


def build_citation_bundle(evidence_pack: dict[str, Any]) -> list[dict[str, Any]]:
    citations = []
    seen: set[str] = set()
    for url in evidence_pack.get("official_sources", []) + evidence_pack.get("cross_sources", []):
        if url in seen:
            continue
        seen.add(url)
        index = len(citations) + 1
        citations.append({"index": index, "url": url})
    return citations[:8]
