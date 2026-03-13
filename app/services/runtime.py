from __future__ import annotations

import importlib.util
import json
import re
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Optional

from ..config import WRITER_BACKEND_POLICY
from ..db import get_connection
from .editorial_ai import OpenAIEditorialClient

_MODEL_CACHE: dict[tuple[str, str | None], tuple[Any, Any]] = {}
_MODEL_CACHE_LOCK = Lock()

_CATEGORY_LABELS = {
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

_OUTLINE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "headline": {"type": "string"},
        "lead": {"type": "string"},
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "section_key": {"type": "string"},
                    "title": {"type": "string"},
                    "angle": {"type": "string"},
                },
                "required": ["section_key", "title", "angle"],
            },
        },
    },
    "required": ["headline", "lead", "sections"],
}
_SECTION_BRIEF_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "angle": {"type": "string"},
        "fact_chain": {"type": "array", "items": {"type": "string"}},
        "why_it_matters": {"type": "array", "items": {"type": "string"}},
        "judgement": {"type": "string"},
        "market_link": {"type": "string"},
        "risk_watch": {"type": "array", "items": {"type": "string"}},
        "confidence_boundary": {"type": "string"},
    },
    "required": [
        "angle",
        "fact_chain",
        "why_it_matters",
        "judgement",
        "market_link",
        "risk_watch",
        "confidence_boundary",
    ],
}


@dataclass
class RuntimeModelConfig:
    artifact_id: Optional[str]
    base_model: str
    adapter_path: Optional[str]
    status: str
    source: str
    backend: str


def _clean_snippet(value: Any) -> str:
    text = str(value or "")
    text = text.replace("\u3000", " ").replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _clean_list(items: Any, limit: int = 4, min_length: int = 6) -> list[str]:
    results: list[str] = []
    for item in items or []:
        cleaned = _clean_snippet(item)
        if len(cleaned) < min_length:
            continue
        if cleaned not in results:
            results.append(cleaned)
        if len(results) >= limit:
            break
    return results


def _clean_style_guidance(items: Any, limit: int = 4) -> list[str]:
    results: list[str] = []
    for item in items or []:
        cleaned = _clean_snippet(item)
        if len(cleaned) < 8:
            continue
        lowered = cleaned.lower()
        if "http://" in lowered or "https://" in lowered:
            continue
        if "相关链接" in cleaned or "每周订阅内容" in cleaned:
            continue
        if re.match(r"^[（(]?\d+[）)]", cleaned):
            continue
        if sum(ch.isdigit() for ch in cleaned) > max(3, len(cleaned) // 5):
            continue
        if cleaned not in results:
            results.append(cleaned)
        if len(results) >= limit:
            break
    return results


def _format_market_snapshot(market_snapshot: list[dict[str, Any]], limit: int = 8) -> str:
    if not market_snapshot:
        return "暂无可用行情快照。"
    lines = []
    for snapshot in market_snapshot[:limit]:
        lines.append(
            f"{snapshot['display_name']}：周涨跌 {snapshot['weekly_return']}%，年内 {snapshot['ytd_return']}%，最新收盘 {snapshot['latest_close']}"
        )
    return "\n".join(lines)


def _format_evidence_pack(section_title: str, evidence_pack: dict[str, Any]) -> str:
    parts = [f"主题：{section_title}"]
    summary = _clean_snippet(evidence_pack.get("event_summary"))
    if summary:
        parts.append(f"事件摘要：{summary}")
    key_facts = _clean_list(evidence_pack.get("key_facts"), limit=5, min_length=4)
    if key_facts:
        parts.append("关键事实：" + "；".join(key_facts))
    entities = _clean_list(evidence_pack.get("entities"), limit=8, min_length=2)
    if entities:
        parts.append("相关实体：" + "、".join(entities))
    timeline = []
    for item in evidence_pack.get("timeline", [])[:4]:
        if isinstance(item, dict):
            title = _clean_snippet(item.get("title"))
            published_at = _clean_snippet(item.get("published_at"))
            url = _clean_snippet(item.get("url"))
            timeline_item = " | ".join(part for part in [published_at, title, url] if part)
        else:
            timeline_item = _clean_snippet(item)
        if timeline_item:
            timeline.append(timeline_item)
    if timeline:
        parts.append("时间线：" + "；".join(timeline))
    official_sources = _clean_list(evidence_pack.get("official_sources"), limit=3, min_length=8)
    cross_sources = _clean_list(evidence_pack.get("cross_sources"), limit=4, min_length=8)
    if official_sources:
        parts.append("官方来源：" + "；".join(official_sources))
    if cross_sources:
        parts.append("交叉来源：" + "；".join(cross_sources))
    impact_paths = _clean_list(evidence_pack.get("impact_paths"), limit=4, min_length=6)
    if impact_paths:
        parts.append("影响路径：" + "；".join(impact_paths))
    boundaries = _clean_list(evidence_pack.get("allowed_conclusions"), limit=3, min_length=6)
    if boundaries:
        parts.append("允许的结论边界：" + "；".join(boundaries))
    forbidden = _clean_list(
        evidence_pack.get("forbidden_judgements") or evidence_pack.get("forbidden_claims"),
        limit=3,
        min_length=6,
    )
    if forbidden:
        parts.append("禁止越界判断：" + "；".join(forbidden))
    confidence_note = _clean_snippet(evidence_pack.get("confidence_note"))
    if confidence_note:
        parts.append("证据强度：" + confidence_note)
    return "\n".join(parts)


def _style_blueprint(style_profile: dict[str, Any]) -> dict[str, Any]:
    stats = style_profile.get("stats") or {}
    blueprint = stats.get("blueprint") or {}
    return {
        "structure": blueprint.get("structure") or {},
        "market_review": blueprint.get("market_review") or {},
        "voice": blueprint.get("voice") or {},
    }


def _market_focus_labels(style_profile: dict[str, Any]) -> str:
    blueprint = _style_blueprint(style_profile)
    focuses = blueprint.get("market_review", {}).get("focuses") or []
    label_map = {
        "valuation": "估值与盈利兑现",
        "rates": "利率路径与宏观预期",
        "tech": "科技龙头与AI主线",
        "macro": "政策边界、美元与风险偏好",
    }
    resolved = [label_map.get(item, item) for item in focuses[:4]]
    return "、".join(resolved) or "估值、利率路径、风险偏好和科技主线"


def _preferred_structure(style_profile: dict[str, Any]) -> str:
    blueprint = _style_blueprint(style_profile)
    structure = blueprint.get("structure", {})
    flow = structure.get("topic_flow") or "主题标题 -> 相关链接 -> 评述"
    return f"优先采用“{flow}”，主题写完后单列市场评述，最后用本周总判断收束。"


def _worldview_summary(style_profile: dict[str, Any]) -> str:
    notes = _clean_style_guidance(style_profile.get("worldview_notes"), limit=4)
    focus_parts: list[str] = []
    if any(token in " ".join(notes) for token in ("估值", "现金流", "预期差")):
        focus_parts.append("估值锚和现金流质量")
    if any(token in " ".join(notes) for token in ("利率", "通胀", "美元", "流动性")):
        focus_parts.append("利率路径与美元流动性")
    if any(token in " ".join(notes) for token in ("科技", "AI", "芯片", "资本开支")):
        focus_parts.append("科技主线与资本开支")
    if any(token in " ".join(notes) for token in ("政策", "地缘", "中美", "台海", "俄乌", "伊朗")):
        focus_parts.append("政策边界与地缘风险")
    if not focus_parts:
        focus_parts.append(_market_focus_labels(style_profile))
    return f"重点看{'、'.join(focus_parts[:4])}如何共同影响资产定价，而不是追逐一周噪音。"


def _evidence_boundary(evidence_pack: dict[str, Any]) -> str:
    official_count = len(evidence_pack.get("official_sources", []) or [])
    cross_count = len(evidence_pack.get("cross_sources", []) or [])
    if official_count + cross_count >= 5:
        return "就现有证据看，方向判断可以比普通新闻更往前走一步，但仍然不把结论下满。"
    if official_count + cross_count >= 2:
        return "现有材料已经足够支持方向判断，但更激进的推演还要继续等后续证据。"
    return "目前证据还偏薄，所以判断只能先停在趋势和风险提示这一层。"


def _impact_frame(category: Optional[str]) -> str:
    mapping = {
        "us_politics": "它会不会继续改写美国政策边界、盟友关系和市场对政治风险的定价",
        "china_politics": "它会不会继续改变中国的政策姿态、对外关系和风险偏好",
        "china_macro": "它会不会继续影响增长预期、政策空间和中国资产的定价锚",
        "us_economy": "它会不会继续改变利率路径、增长预期和美元资产的估值框架",
        "us_technology": "它会不会继续影响科技资本开支、行业竞争格局和高估值资产的容忍度",
        "cross_strait": "它会不会继续抬升台海风险溢价，并改变中美与区域安全预期",
        "russia_ukraine": "它会不会继续影响欧洲安全、能源价格和全球风险偏好",
        "middle_east": "它会不会继续影响中东局势、能源链条和全球风险偏好",
        "global_geopolitics": "它会不会继续改变大国博弈、关税制裁和全球风险资产的定价逻辑",
    }
    return mapping.get(category, "它会不会继续改变政策边界、市场预期和资产定价")


def _extract_json_block(text: str) -> Optional[dict[str, Any]]:
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return None
    candidate = match.group(0)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _style_bundle(style_profile: dict[str, Any]) -> str:
    voice_notes = _clean_style_guidance(style_profile.get("voice_notes"), limit=4)
    worldview = _worldview_summary(style_profile)
    structure = _preferred_structure(style_profile)
    banned = _clean_style_guidance(style_profile.get("banned_phrases"), limit=6)
    parts = []
    if voice_notes:
        parts.append("说话方式：" + "；".join(voice_notes))
    if worldview:
        parts.append("三观与判断框架：" + worldview)
    if structure:
        parts.append("结构要求：" + structure)
    if banned:
        parts.append("禁用表达：" + "；".join(banned))
    return "\n".join(parts)


class LocalWriterRuntime:
    def __init__(
        self,
        model_name: Optional[str] = None,
        adapter_path: Optional[str] = None,
        force_backend: Optional[str] = None,
    ) -> None:
        self._openai = OpenAIEditorialClient()
        self._resolved = self._resolve_model(model_name, adapter_path, force_backend=force_backend)
        self.model_name = self._resolved.base_model

    def available(self) -> bool:
        return self._resolved.backend in {"mlx", "openai"}

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self._resolved.backend,
            "artifact_id": self._resolved.artifact_id,
            "base_model": self._resolved.base_model,
            "adapter_path": self._resolved.adapter_path,
            "status": self._resolved.status,
            "source": self._resolved.source,
        }

    def _resolve_model(
        self,
        model_name: Optional[str],
        adapter_path: Optional[str],
        force_backend: Optional[str] = None,
    ) -> RuntimeModelConfig:
        if model_name:
            resolved_adapter = str(Path(adapter_path).resolve()) if adapter_path else None
            return RuntimeModelConfig(
                artifact_id=None,
                base_model=model_name,
                adapter_path=resolved_adapter,
                status="manual",
                source="manual",
                backend=force_backend or ("mlx" if self._mlx_available() else "heuristic"),
            )

        if force_backend == "openai" and self._openai.available():
            return RuntimeModelConfig(
                artifact_id=None,
                base_model=self._openai.model,
                adapter_path=None,
                status="api",
                source="openai-api",
                backend="openai",
            )
        if force_backend == "heuristic":
            return RuntimeModelConfig(
                artifact_id=None,
                base_model="heuristic-local-writer",
                adapter_path=None,
                status="manual",
                source="manual",
                backend="heuristic",
            )

        with get_connection() as connection:
            row = connection.execute(
                """
                SELECT id, base_model, adapter_path, status, is_production
                FROM model_artifacts
                WHERE role = 'writer' AND status IN ('production', 'trained')
                ORDER BY is_production DESC, created_at DESC
                LIMIT 1
                """
            ).fetchone()

        if WRITER_BACKEND_POLICY == "openai_first" and self._openai.available():
            return RuntimeModelConfig(
                artifact_id=row["id"] if row else None,
                base_model=self._openai.model,
                adapter_path=None,
                status="api",
                source="openai-api",
                backend="openai",
            )

        if not row:
            return RuntimeModelConfig(
                artifact_id=None,
                base_model="heuristic-local-writer",
                adapter_path=None,
                status="not_set",
                source="none",
                backend=force_backend or "heuristic",
            )

        resolved_adapter = row["adapter_path"]
        if resolved_adapter:
            resolved_adapter = str(Path(resolved_adapter).resolve())
        if (
            not self._mlx_available()
            or (resolved_adapter and not Path(resolved_adapter).exists())
            or self._is_placeholder_adapter(resolved_adapter)
        ):
            return RuntimeModelConfig(
                artifact_id=row["id"],
                base_model=row["base_model"],
                adapter_path=resolved_adapter,
                status=row["status"],
                source="production" if row["is_production"] else "latest-trained",
                backend=force_backend or "heuristic",
            )

        return RuntimeModelConfig(
            artifact_id=row["id"],
            base_model=row["base_model"],
            adapter_path=resolved_adapter,
            status=row["status"],
            source="production" if row["is_production"] else "latest-trained",
            backend=force_backend or "mlx",
        )

    def _mlx_available(self) -> bool:
        return importlib.util.find_spec("mlx_lm") is not None

    def _is_placeholder_adapter(self, adapter_path: Optional[str]) -> bool:
        if not adapter_path:
            return False
        adapter_dir = Path(adapter_path)
        if not adapter_dir.exists() or not adapter_dir.is_dir():
            return False
        adapter_weights = adapter_dir / "adapters.safetensors"
        if adapter_weights.exists():
            return False
        readme_path = adapter_dir / "README.txt"
        if not readme_path.exists():
            return False
        try:
            readme_text = readme_path.read_text(encoding="utf-8").lower()
        except OSError:
            return False
        return "placeholder adapter" in readme_text or "replace with real training output" in readme_text

    def _load_components(self) -> tuple[Any, Any]:
        cache_key = (self._resolved.base_model, self._resolved.adapter_path)
        with _MODEL_CACHE_LOCK:
            cached = _MODEL_CACHE.get(cache_key)
            if cached is not None:
                return cached
            from mlx_lm import load

            model, tokenizer = load(
                self._resolved.base_model,
                adapter_path=self._resolved.adapter_path,
            )
            _MODEL_CACHE[cache_key] = (model, tokenizer)
            return model, tokenizer

    def _chat_generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int,
        temp: float,
        top_p: float = 0.9,
    ) -> str:
        model, tokenizer = self._load_components()
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=make_sampler(temp=temp, top_p=top_p),
            verbose=False,
        )
        return self._post_process(response)

    def _post_process(self, text: str) -> str:
        cleaned = text.strip()
        cleaned = re.sub(r"^```(?:json|markdown)?", "", cleaned, flags=re.I).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
        cleaned = re.sub(r"<\\/?think>", "", cleaned, flags=re.I).strip()
        cleaned = re.sub(r"<\\|.*?\\|>", "", cleaned, flags=re.S).strip()
        return cleaned

    def _openai_outline(self, issue_title: str, topics: list[dict[str, Any]], style_profile: dict[str, Any]) -> Optional[dict[str, Any]]:
        topic_lines = "\n".join(
            f"- {topic['title']}（{_CATEGORY_LABELS.get(topic['category'], topic['category'])}）: {topic.get('evidence_pack', {}).get('event_summary', '')}"
            for topic in topics
        )
        try:
            parsed = self._openai.create_json(
                instructions=(
                    "你是作者本人的中文研究编辑。"
                    "请把已选主题编排成成熟付费周报的提纲，开头必须有主判断，章节角度必须有区分。"
                    "不要写成媒体汇总，不要平均用力。"
                ),
                input_text=(
                    f"标题：{issue_title}\n"
                    f"{_style_bundle(style_profile)}\n\n"
                    f"已选主题：\n{topic_lines}"
                ),
                schema_name="weekly_outline",
                schema=_OUTLINE_SCHEMA,
                max_output_tokens=700,
            )
        except Exception:
            return None
        sections = parsed.get("sections") or []
        if not isinstance(sections, list):
            return None
        return {
            "headline": _clean_snippet(parsed.get("headline")) or issue_title,
            "lead": _clean_snippet(parsed.get("lead")),
            "sections": [
                {
                    "section_key": _clean_snippet(section.get("section_key")) or f"topic-{index + 1}",
                    "title": _clean_snippet(section.get("title")) or topics[index]["title"],
                    "angle": _clean_snippet(section.get("angle")) or f"围绕{topics[index]['title']}展开。",
                }
                for index, section in enumerate(sections[: len(topics)])
            ],
        }

    def _openai_section_brief(
        self,
        section_title: str,
        evidence_pack: dict[str, Any],
        style_profile: dict[str, Any],
        market_snapshot: Optional[list[dict[str, Any]]] = None,
        category: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        evidence_text = _format_evidence_pack(section_title, evidence_pack)
        market_text = _format_market_snapshot(market_snapshot or [], limit=5)
        try:
            return self._openai.create_json(
                instructions=(
                    "你是作者本人的事实编辑。"
                    "先从证据里提炼事实链和真正重要的矛盾，再给出有限但鲜明的判断。"
                    "不要把材料外信息混进来，也不要输出空泛套话。"
                ),
                input_text=(
                    f"章节标题：{section_title}\n"
                    f"主题类别：{_CATEGORY_LABELS.get(category or '', category or '综合')}\n"
                    f"{_style_bundle(style_profile)}\n\n"
                    f"证据包：\n{evidence_text}\n\n"
                    f"市场背景：\n{market_text}"
                ),
                schema_name="section_brief",
                schema=_SECTION_BRIEF_SCHEMA,
                max_output_tokens=800,
            )
        except Exception:
            return None

    def _openai_section_candidates(
        self,
        section_title: str,
        evidence_pack: dict[str, Any],
        style_profile: dict[str, Any],
        market_snapshot: Optional[list[dict[str, Any]]] = None,
        category: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        brief = self._openai_section_brief(section_title, evidence_pack, style_profile, market_snapshot, category)
        if not brief:
            return self._heuristic_candidates(section_title, evidence_pack, style_profile, market_snapshot, category)
        evidence_text = _format_evidence_pack(section_title, evidence_pack)
        variants = [
            "开头先压住事实主线，再把观点推出来。",
            "从长期框架切入，强调为什么这不是一次性噪音。",
            "把市场或政策传导写得更锋利，但不要越出证据边界。",
        ]
        candidates: list[dict[str, Any]] = []
        for index, variant in enumerate(variants, start=1):
            prompt = (
                f"章节标题：{section_title}\n"
                f"主题类别：{_CATEGORY_LABELS.get(category or '', category or '综合')}\n"
                f"{_style_bundle(style_profile)}\n\n"
                f"证据包：\n{evidence_text}\n\n"
                f"编辑简报：\n"
                f"角度：{brief.get('angle', '')}\n"
                f"事实链：{'；'.join(brief.get('fact_chain', []))}\n"
                f"为什么重要：{'；'.join(brief.get('why_it_matters', []))}\n"
                f"核心判断：{brief.get('judgement', '')}\n"
                f"市场映射：{brief.get('market_link', '')}\n"
                f"需要盯住：{'；'.join(brief.get('risk_watch', []))}\n"
                f"边界：{brief.get('confidence_boundary', '')}\n\n"
                "任务：写这一节中“评述：”后面的正文。"
                "必须先交代事实，再展开判断，再落到政策、产业、地缘或资产定价影响。"
                "允许明确使用“我认为 / 我的判断是 / 更重要的是”。"
                "不要写标题，不要写来源列表，不要出现模板腔。"
                "如果证据只支持有限判断，要把边界说清楚。"
                f"{variant} 篇幅 320 到 560 字。"
            )
            try:
                text = self._openai.create_text(
                    instructions="你是观点鲜明、但严格尊重证据边界的中文财经与国际政治订阅作者。",
                    input_text=prompt,
                    max_output_tokens=900,
                )
            except Exception:
                return self._heuristic_candidates(section_title, evidence_pack, style_profile, market_snapshot, category)
            candidates.append({"text": self._post_process(text), "sampling_note": f"openai_candidate_{index}"})
        return candidates

    def _heuristic_outline(self, issue_title: str, topics: list[dict[str, Any]], style_profile: dict[str, Any]) -> dict[str, Any]:
        topic_titles = [topic["title"] for topic in topics]
        worldview = _worldview_summary(style_profile)
        structure_note = _preferred_structure(style_profile)
        return {
            "headline": issue_title,
            "lead": (
                f"这一周真正值得放在一起看的，不只是 {'；'.join(topic_titles[:3])} 这些单条新闻。"
                f"我更关注的是这些主题背后共同指向的判断框架：{worldview}。"
                f"{structure_note}"
            ),
            "sections": [
                {
                    "section_key": f"topic-{index + 1}",
                    "title": topic["title"],
                    "angle": f"从{_CATEGORY_LABELS.get(topic['category'], topic['category'])}角度评估其对政策、预期与资产定价的影响。",
                }
                for index, topic in enumerate(topics)
            ],
        }

    def _heuristic_candidates(
        self,
        section_title: str,
        evidence_pack: dict[str, Any],
        style_profile: dict[str, Any],
        market_snapshot: Optional[list[dict[str, Any]]] = None,
        category: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        facts = _clean_list(evidence_pack.get("key_facts"), limit=4, min_length=4)
        event_summary = _clean_snippet(evidence_pack.get("event_summary")) or section_title
        worldview = _worldview_summary(style_profile)
        market_hint = market_snapshot[0]["display_name"] if market_snapshot else "风险资产"
        conclusion_boundary = _evidence_boundary(evidence_pack)
        fact_chain = "；".join(facts[:3]) or event_summary
        impact_frame = _impact_frame(category)
        opening = facts[0] if facts else event_summary

        variants = [
            (
                f"{opening}。如果把这一周出现的线索放在一起看，{fact_chain} 已经不是孤立消息。"
                f"我认为真正重要的不是新闻本身，而是 {impact_frame}。"
                f"{conclusion_boundary}"
            ),
            (
                f"{event_summary}。把时间线拉长看，它更像是同一条主线的继续累积，而不是一次性噪音。"
                f"我的判断是，接下来最值得盯住的是它会不会继续传导到政策预期、企业行为和 {market_hint} 的风险偏好。"
                f"{conclusion_boundary}"
            ),
            (
                f"{event_summary}。更重要的是，{fact_chain} 放在一起已经足以说明方向开始变得清楚。"
                f"从作者一贯的框架看，{worldview}，所以这里最需要警惕的不是消息不够多，而是市场和舆论过早把结论交易得太满。"
                f"{conclusion_boundary}"
            ),
        ]

        candidates = []
        for index, variant in enumerate(variants):
            if market_snapshot:
                variant += f" 如果这条线继续发酵，最终会映射到 {market_hint} 以及更广泛的资产定价。"
            candidates.append({"text": variant, "sampling_note": f"heuristic_candidate_{index + 1}"})
        return candidates

    def generate_outline(self, issue_title: str, topics: list[dict[str, Any]], style_profile: dict[str, Any]) -> dict[str, Any]:
        if self._resolved.backend == "openai":
            parsed = self._openai_outline(issue_title, topics, style_profile)
            if parsed:
                return parsed
            return self._heuristic_outline(issue_title, topics, style_profile)

        if not self.available():
            return self._heuristic_outline(issue_title, topics, style_profile)

        worldview_notes = _worldview_summary(style_profile)
        preferred_patterns = _preferred_structure(style_profile)
        topic_lines = "\n".join(
            f"- {topic['title']}（{_CATEGORY_LABELS.get(topic['category'], topic['category'])}）"
            for topic in topics
        )
        prompt = (
            f"请为题为《{issue_title}》的周报生成内部提纲。\n"
            f"已选主题：\n{topic_lines}\n\n"
                f"风格取向：{worldview_notes or '重视估值、现金流和政策边界。'}\n"
            f"结构偏好：{preferred_patterns}\n\n"
            "请只输出 JSON，不要解释，格式如下：\n"
            '{"headline":"...","lead":"...","sections":[{"section_key":"topic-1","title":"...","angle":"..."},{"section_key":"topic-2","title":"...","angle":"..."}]}'
        )
        try:
            raw = self._chat_generate(
                "你是作者本人的研究助理，只输出符合要求的 JSON，提纲必须像成熟付费周报而不是媒体资讯汇总。",
                prompt,
                max_tokens=280,
                temp=0.2,
            )
            parsed = _extract_json_block(raw)
            if not parsed:
                raise ValueError("outline JSON parse failed")
            sections = parsed.get("sections") or []
            if not parsed.get("lead") or not isinstance(sections, list):
                raise ValueError("outline missing fields")
            return {
                "headline": parsed.get("headline") or issue_title,
                "lead": _clean_snippet(parsed.get("lead")),
                "sections": [
                    {
                        "section_key": _clean_snippet(section.get("section_key")) or f"topic-{index + 1}",
                        "title": _clean_snippet(section.get("title")) or topics[index]["title"],
                        "angle": _clean_snippet(section.get("angle")) or f"围绕{topics[index]['title']}展开。",
                    }
                    for index, section in enumerate(sections[: len(topics)])
                ],
            }
        except Exception:
            return self._heuristic_outline(issue_title, topics, style_profile)

    def generate_candidates(
        self,
        section_title: str,
        evidence_pack: dict[str, Any],
        style_profile: dict[str, Any],
        market_snapshot: Optional[list[dict[str, Any]]] = None,
        category: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        if self._resolved.backend == "openai":
            return self._openai_section_candidates(section_title, evidence_pack, style_profile, market_snapshot, category)

        if not self.available():
            return self._heuristic_candidates(section_title, evidence_pack, style_profile, market_snapshot, category)

        worldview = _worldview_summary(style_profile)
        banned_phrases = "、".join(_clean_list(style_profile.get("banned_phrases"), limit=6, min_length=2))
        evidence_text = _format_evidence_pack(section_title, evidence_pack)
        market_text = _format_market_snapshot(market_snapshot or [], limit=4)
        structure_text = _preferred_structure(style_profile)
        instructions = [
            "先交代事实，再解释为什么重要，再给出作者判断，最后落到政策、产业、地缘或资产定价影响。",
            "允许明确写出“我认为 / 我的判断是 / 更重要的是”，但判断必须紧贴证据边界，不要空喊立场。",
            "不要像媒体快讯；要像作者给付费读者写的评述段落，观点要鲜明，边界也要鲜明。",
        ]
        candidates: list[dict[str, Any]] = []
        for index, instruction in enumerate(instructions, start=1):
            prompt = (
                f"{evidence_text}\n\n"
                f"相关市场背景：\n{market_text}\n\n"
                f"作者立场与取向：{worldview or '重视估值、现金流和政策边界。'}\n"
                f"历史结构偏好：{structure_text}\n"
                f"主题类别：{_CATEGORY_LABELS.get(category or '', category or '综合')}\n"
                f"禁用表达：{banned_phrases or '无'}\n\n"
                f"写作任务：写《{section_title}》这一节中紧跟在“评述：”后面的正文，不要写标题，不要写来源列表，不要出现“作为AI”。"
                "必须严格局限于当前 evidence pack 里的事实、实体和时间线，不要引入其他主题的素材。"
                "如果当前证据没有出现房地产、LPR、按揭、导购、开箱、价格比较等信息，就不要自行补写这些内容。"
                f"要求：{instruction} 如果来源不足，要明确写出只能做有限判断。篇幅 280 到 520 字。"
            )
            try:
                text = self._chat_generate(
                    "你是有鲜明判断力的中文财经与国际政治订阅作者，写作要求是先事实后判断，但绝不写成温吞的均衡报道。",
                    prompt,
                    max_tokens=360,
                    temp=0.3 + (index - 1) * 0.08,
                )
            except Exception:
                return self._heuristic_candidates(section_title, evidence_pack, style_profile, market_snapshot)
            candidates.append(
                {
                    "text": text,
                    "sampling_note": f"mlx_candidate_{index}",
                }
            )
        return candidates

    def generate_market_review(self, market_snapshot: list[dict[str, Any]], style_profile: dict[str, Any]) -> str:
        worldview = _worldview_summary(style_profile)
        market_focus = _market_focus_labels(style_profile)
        if not market_snapshot:
            return "本周暂无完整市场快照，因此市场评述只保留框架：优先盯估值、利率路径和科技主线，不在缺少数据时硬下结论。"
        if self._resolved.backend == "openai":
            prompt = (
                f"{_style_bundle(style_profile)}\n\n"
                f"市场快照：\n{_format_market_snapshot(market_snapshot, limit=12)}\n\n"
                f"历史市场评述重心：{market_focus}\n"
                "任务：写正式周报里的“市场评述”。"
                "必须覆盖 S&P 500、Nasdaq 100、VIX、US 10Y、DXY，并说明科技主线和估值框架的关系。"
                "不要流水账，不要平铺涨跌，要写出作者本周真正的主判断。"
                "允许明确写“我认为”或“更重要的是”，但不能脱离给定数据。"
                "篇幅 480 到 820 字。"
            )
            try:
                return self._post_process(
                    self._openai.create_text(
                        instructions="你是中文财经订阅作者，擅长把市场数据压成有判断的周度评述。",
                        input_text=prompt,
                        max_output_tokens=1100,
                    )
                )
            except Exception:
                return self.generate_market_review([], style_profile)
        if not self.available():
            worldview_fallback = worldview or "真正重要的是长期现金流和估值锚，而不是一周涨跌。"
            snapshot_map = {item["display_name"]: item for item in market_snapshot}
            spx = snapshot_map.get("S&P 500")
            ndx = snapshot_map.get("Nasdaq 100")
            vix = snapshot_map.get("VIX")
            us10y = snapshot_map.get("US 10Y")
            dxy = snapshot_map.get("DXY")
            opening = "本周市场里最值得看的，不是指数涨跌本身，而是市场在用什么框架给未来几个月定价。"
            performance = []
            if spx:
                performance.append(f"S&P 500 本周 {spx['weekly_return']}%，年内 {spx['ytd_return']}%")
            if ndx:
                performance.append(f"Nasdaq 100 本周 {ndx['weekly_return']}%，年内 {ndx['ytd_return']}%")
            if vix:
                performance.append(f"VIX 当前一周变化 {vix['weekly_return']}%")
            if us10y:
                performance.append(f"US 10Y 一周变化 {us10y['weekly_return']}%")
            if dxy:
                performance.append(f"DXY 一周变化 {dxy['weekly_return']}%")
            interpretation = (
                "如果纳指继续强于标普，说明市场仍在把筹码压在科技龙头和 AI 主线；"
                "如果利率和美元没有出现决定性拐点，高估值资产就很难真正进入无差别扩张阶段。"
            )
            return (
                f"{opening}{'；'.join(performance)}。"
                f"我认为这周的市场评述应该主要盯住 {market_focus}。"
                f"{interpretation} {worldview_fallback}"
            )

        prompt = (
            "请根据以下市场快照，写一段正式周报里的“市场评述”。\n\n"
            f"{_format_market_snapshot(market_snapshot, limit=12)}\n\n"
            f"作者取向：{worldview or '重视长期现金流、估值锚和政策边界。'}\n"
            f"历史市场评述侧重点：{market_focus}。\n"
            "要求：覆盖 S&P 500、Nasdaq 100、VIX、US 10Y、DXY，并带到科技观察池；"
            "不要逐条报菜名，不要空洞乐观或悲观，要像成熟投资通讯。"
            "必须有明确主判断，最好能写出“我认为”或“更重要的是”，但不能脱离数据乱下结论。"
            "只能根据上面的市场快照写，不要引入中国降息、LPR、房贷、按揭、房地产、再贷款等未提供的信息。"
            "篇幅 450 到 800 字。"
        )
        try:
            return self._chat_generate(
                "你是重视估值框架和事实边界、同时观点鲜明的中文财经作者。",
                prompt,
                max_tokens=520,
                temp=0.28,
            )
        except Exception:
            return self.generate_market_review([], style_profile)

    def generate_conclusion(self, topics: list[dict[str, Any]], style_profile: dict[str, Any]) -> str:
        categories = "、".join(_CATEGORY_LABELS.get(topic["category"], topic["category"]) for topic in topics[:4])
        worldview = _worldview_summary(style_profile)
        if self._resolved.backend == "openai":
            topic_lines = "\n".join(
                f"- {topic['title']}：{topic.get('evidence_pack', {}).get('event_summary', '')}"
                for topic in topics[:5]
            )
            prompt = (
                f"{_style_bundle(style_profile)}\n\n"
                f"本期主题类别：{categories}\n"
                f"主题摘要：\n{topic_lines}\n\n"
                "任务：写“本周总判断”。"
                "不要重复前文事实，而是把本周几条线索收束成作者的总判断、风险排序和下一周的观察重点。"
                "需要有一句明确主判断，让立场清晰可感，但仍然保持边界感。"
                "篇幅 220 到 360 字。"
            )
            try:
                return self._post_process(
                    self._openai.create_text(
                        instructions="你是成熟、克制但立场清晰的中文财经与国际政治通讯作者。",
                        input_text=prompt,
                        max_output_tokens=480,
                    )
                )
            except Exception:
                worldview_fallback = worldview or "在高波动环境里，最重要的是不被噪音带偏。"
                return (
                    f"把这一周的主题放在一起看，{categories or '这些主线'}并不是彼此割裂。"
                    f"我的判断是，真正值得继续跟踪的仍然是那些会改变预期差和资产定价的大变量。"
                    f" {worldview_fallback}"
                )
        if not self.available():
            worldview_fallback = worldview or "在高波动环境里，最重要的是不被噪音带偏。"
            return (
                f"把这一周的主题放在一起看，{categories}并不是彼此割裂。"
                f"我的判断是，真正值得继续跟踪的仍然是那些会改变预期差和资产定价的大变量。"
                f" {worldview_fallback}"
            )

        prompt = (
            f"本期涉及的主题类别包括：{categories}。\n"
            f"作者取向：{worldview or '重视估值、现金流、政策边界，不追逐短期噪音。'}\n"
            "请写“本周总判断”这一节，不重复前文事实，不喊口号，要像成熟付费订阅最后的收束段。"
            "需要有一句明确主判断，例如“我的判断是”或“更重要的是”，让作者立场清晰可感。"
            "只能围绕本期主题类别和长期投资框架收束，不要引入中国降息、LPR、房贷、按揭、房地产、再贷款等无关旧素材。"
            "篇幅 180 到 320 字。"
        )
        try:
            return self._chat_generate(
                "你是成熟克制但观点鲜明的中文财经付费通讯作者。",
                prompt,
                max_tokens=220,
                temp=0.25,
            )
        except Exception:
            worldview_fallback = worldview or "在高波动环境里，最重要的是不被噪音带偏。"
            return (
                f"最后把这一周的主题放在一起看，{categories}并不是彼此割裂。"
                f" 它们共同决定了未来几个季度的预期差在哪里，而投资的关键仍然是 {worldview_fallback}"
            )
