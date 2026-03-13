from __future__ import annotations

import shutil
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from app.config import DATA_DIR, DB_PATH
from app.db import get_connection, init_db
from app.main import app
from app.services.critic import build_citation_bundle, score_candidate
from app.services.corpus import latest_style_profile, split_sections
from app.services.runtime import LocalWriterRuntime
from app.services.sources import extract_entities, rebuild_topic_clusters, recommend_topics, seed_sources, topic_preference_overview
from app.utils import compact_whitespace, current_week_key, json_dumps, now_iso


class LocalBriefingStudioTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._backup_root = Path(tempfile.mkdtemp(prefix="local-briefing-studio-tests-"))
        self._data_backup_path = self._backup_root / "data-backup"
        self._download_dir = self._backup_root / "downloads"
        self._download_dir.mkdir(parents=True, exist_ok=True)
        self.addCleanup(self._restore_data_dir)
        if DATA_DIR.exists():
            shutil.move(str(DATA_DIR), str(self._data_backup_path))
        init_db()
        seed_sources()
        self.client = TestClient(app)

    def _restore_data_dir(self) -> None:
        if DATA_DIR.exists():
            shutil.rmtree(DATA_DIR)
        if self._data_backup_path.exists():
            shutil.move(str(self._data_backup_path), str(DATA_DIR))
        shutil.rmtree(self._backup_root, ignore_errors=True)

    def _insert_article(self, title: str, category: str, source_id: str, summary: str, content: str) -> None:
        entities = extract_entities(title, content)
        with get_connection() as connection:
            connection.execute(
                """
                INSERT INTO raw_articles (
                  id, source_id, title, url, published_at, language, category, author,
                  summary, content, content_hash, tags_json, entities_json, trust_level, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uuid.uuid4().hex,
                    source_id,
                    title,
                    f"https://example.com/{uuid.uuid4().hex}",
                    now_iso(),
                    "zh",
                    category,
                    "",
                    summary,
                    content,
                    str(uuid.uuid4()),
                    "[]",
                    json_dumps(entities),
                    "high",
                    now_iso(),
                ),
            )

    def _insert_corpus_doc(self, title: str, sections: list[dict[str, str]]) -> None:
        content = "\n\n".join(section["content"] for section in sections if section.get("content"))
        with get_connection() as connection:
            connection.execute(
                """
                INSERT INTO style_corpus_docs (
                  id, file_name, file_type, imported_at, title, content, sections_json, metadata_json, structure_tags_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uuid.uuid4().hex,
                    f"{title}.txt",
                    "text/plain",
                    now_iso(),
                    title,
                    content,
                    json_dumps(sections),
                    "{}",
                    "[]",
                ),
            )

    def test_corpus_import_builds_profile(self) -> None:
        response = self.client.post(
            "/corpus/import",
            files=[
                (
                    "files",
                    (
                        "sample.txt",
                        (
                            "# 本周观察\n\n"
                            "美国经济这一周最大的变化，是市场重新修正了对利率路径的预期。\n\n"
                            "市场评述\n\n"
                            "标普与纳指的波动提醒我们，估值和现金流始终比情绪更重要。"
                        ).encode("utf-8"),
                        "text/plain",
                    ),
                )
            ],
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["imported_documents"], 1)
        self.assertNotEqual(payload["style_profile_version"], "empty")

        overview = self.client.get("/corpus")
        self.assertEqual(overview.status_code, 200)
        self.assertIn("语料导入", overview.text)

    def test_style_profile_learns_structure_and_stance_from_corpus(self) -> None:
        response = self.client.post(
            "/corpus/import",
            files=[
                (
                    "files",
                    (
                        "sample.txt",
                        (
                            "每周订阅内容20260313\n\n"
                            "这周真正重要的不是单条消息，而是它们会不会改变市场对未来的定价。\n\n"
                            "（1）中美元首通话\n\n"
                            "相关链接：https://example.com/us-china\n\n"
                            "评述：\n\n"
                            "我认为中美之间仍然缺乏互信，但这意味着双方都不愿意让冲突完全失控。\n\n"
                            "（2）台海局势升温\n\n"
                            "相关链接：https://example.com/taiwan\n\n"
                            "评述：\n\n"
                            "更重要的是，台海问题最终会回到军事、政治和市场风险偏好的同一张表上。\n\n"
                            "市场评述\n\n"
                            "我觉得本周市场最重要的不是指数涨跌，而是美债利率和科技估值还能不能继续被同时容忍。\n\n"
                            "本周总判断\n\n"
                            "我的判断是，真正要盯住的是估值、政策边界和地缘政治是否同时收紧。"
                        ).encode("utf-8"),
                        "text/plain",
                    ),
                )
            ],
        )
        self.assertEqual(response.status_code, 200)

        profile = latest_style_profile()
        self.assertTrue(any("相关链接" in item and "评述" in item for item in profile["preferred_patterns"]))
        self.assertTrue(any("我认为" in item or "这意味着" in item for item in profile["voice_notes"] + profile["worldview_notes"]))
        self.assertIn("market_review", profile["stats"]["blueprint"])

    def test_compact_whitespace_preserves_paragraph_breaks(self) -> None:
        normalized = compact_whitespace(
            "每周订阅内容20220619\n\n"
            "大家好！我们一起看看过去一周值得关注的事情：\n\n"
            "（1）美联储大幅加息\n\n"
            "相关链接：https://example.com/abc\n\n"
            "评述：\n\n"
            "美联储周三宣布将基准利率上调75个基点。\n"
            "这一动作意味着流动性继续收紧。\n\n"
            "市场评述\n\n"
            "本周标普和纳指继续震荡。\n"
            "估值约束重新回到市场中心。"
        )
        self.assertIn("\n\n", normalized)
        sections = split_sections(normalized)
        self.assertGreaterEqual(len(sections), 2)
        self.assertEqual(sections[0]["task_type"], "intro")

    def test_issue_generation_feedback_and_training_loop(self) -> None:
        self._insert_article(
            title="Fed signals patience as inflation data stays sticky",
            category="us_economy",
            source_id="fed-press",
            summary="Fed 官员继续强调通胀路径存在反复，利率调整仍取决于更多数据。",
            content="Fed 官员继续强调通胀路径存在反复，利率调整仍取决于更多数据。市场开始重新评估降息节奏。",
        )
        self._insert_article(
            title="Nvidia and cloud demand keep AI capex in focus",
            category="us_technology",
            source_id="techcrunch",
            summary="AI 资本开支仍在上修，市场关注盈利兑现与估值约束。",
            content="AI 资本开支仍在上修，市场关注盈利兑现与估值约束。云计算厂商与芯片企业形成新的预期差。",
        )
        rebuild_topic_clusters(current_week_key())

        topics_response = self.client.get(f"/topics?week={current_week_key()}")
        self.assertEqual(topics_response.status_code, 200)
        topics = topics_response.json()
        self.assertGreaterEqual(len(topics), 1)

        issue_response = self.client.post(
            "/issues",
            json={
                "week_key": current_week_key(),
                "title": f"{current_week_key()} 财经订阅周报",
                "topic_ids": [topic["id"] for topic in topics[:2]],
            },
        )
        self.assertEqual(issue_response.status_code, 200)
        issue_id = issue_response.json()["id"]

        from unittest.mock import patch

        with patch(
            "app.services.generation.refresh_market_snapshot",
            return_value=[
                {"display_name": "S&P 500", "weekly_return": 1.2, "ytd_return": 8.5},
                {"display_name": "Nasdaq 100", "weekly_return": 2.1, "ytd_return": 11.3},
            ],
        ):
            generate_response = self.client.post(f"/issues/{issue_id}/generate", json={"regenerate": True})
        self.assertEqual(generate_response.status_code, 200)
        generated_issue = generate_response.json()
        self.assertGreaterEqual(len(generated_issue["sections"]), 3)
        first_section = generated_issue["sections"][0]

        feedback_response = self.client.post(
            f"/issues/{issue_id}/feedback",
            json={
                "verdict": "publish_ready",
                "notes": "整体可发，只改了第一段口气。",
                "sections": [
                    {
                        "section_id": first_section["id"],
                        "original_text": first_section["content"],
                        "edited_text": first_section["content"] + "\n\n补一句：市场不会永远奖励高估值叙事。",
                        "chosen_text": first_section["content"] + "\n\n补一句：市场不会永远奖励高估值叙事。",
                        "rejected_text": first_section["content"],
                        "reason": "tone_alignment",
                        "factuality": 4,
                        "style": 5,
                        "structure": 4,
                        "publishability": 5,
                        "flags": ["不像我"],
                        "notes": "结尾需要更像作者本人。",
                    }
                ],
            },
        )
        self.assertEqual(feedback_response.status_code, 200)
        self.assertEqual(feedback_response.json()["verdict"], "publish_ready")

        sft_response = self.client.post("/training/sft")
        self.assertEqual(sft_response.status_code, 200)
        artifact_id = sft_response.json()["artifact_id"]
        self.assertIn("focus_axis_counts", sft_response.json()["metrics"])
        self.assertGreaterEqual(sft_response.json()["metrics"]["topic_selection_pair_count"], 1)

        preference_response = self.client.post("/training/preferences")
        self.assertEqual(preference_response.status_code, 200)
        self.assertGreaterEqual(preference_response.json()["metrics"]["pair_count"], 1)
        self.assertIn("axis_counts", preference_response.json()["metrics"])
        self.assertIn("voice_alignment", preference_response.json()["metrics"]["axis_counts"])

        promote_response = self.client.post("/models/promote", json={"artifact_id": artifact_id})
        self.assertEqual(promote_response.status_code, 200)
        self.assertEqual(promote_response.json()["id"], artifact_id)

        eval_response = self.client.post("/eval/run")
        self.assertEqual(eval_response.status_code, 200)
        summary = eval_response.json()["summary"]
        self.assertGreaterEqual(summary["avg_publishability"], 5)
        self.assertGreaterEqual(summary["voice_alignment_score"], 5)
        self.assertGreaterEqual(summary["insight_progression_score"], 4)

        export_response = self.client.get(f"/issues/{issue_id}/export?fmt=md")
        self.assertEqual(export_response.status_code, 200)
        self.assertIn("相关链接：", export_response.text)
        self.assertIn("评述：", export_response.text)
        self.assertIn("本周总判断", export_response.text)

    def test_recommended_topics_penalize_weak_consumer_angles(self) -> None:
        self._insert_article(
            title="Where to buy the new iPhone 17E",
            category="us_technology",
            source_id="the-verge",
            summary="A preorder and price guide covering where shoppers can buy the new iPhone 17E.",
            content="This guide explains where to buy the new iPhone 17E, preorder timing, price tiers, and carrier deals.",
        )
        self._insert_article(
            title="Google wraps up $32B acquisition of cloud cybersecurity startup Wiz",
            category="us_technology",
            source_id="techcrunch",
            summary="Google completes its Wiz deal to deepen cloud security and enterprise software reach.",
            content="Google completed the Wiz acquisition, reinforcing cloud security, enterprise software positioning, and AI infrastructure ambitions.",
        )

        rebuild_topic_clusters(current_week_key())
        recommended = recommend_topics(current_week_key(), min_count=1, max_count=2)

        self.assertEqual(len(recommended), 2)
        self.assertEqual(recommended[0]["title"], "Google wraps up $32B acquisition of cloud cybersecurity startup Wiz")
        weak_topic = next(topic for topic in recommended if topic["title"] == "Where to buy the new iPhone 17E")
        self.assertIn("疑似消费/弱选题，已自动降权", weak_topic["recommendation_reasons"])
        self.assertLess(weak_topic["recommendation_score"], recommended[0]["recommendation_score"])

    def test_create_issue_can_generate_immediately(self) -> None:
        self._insert_article(
            title="Fed signals patience as inflation data stays sticky",
            category="us_economy",
            source_id="fed-press",
            summary="Fed 官员继续强调通胀路径存在反复，利率调整仍取决于更多数据。",
            content="Fed 官员继续强调通胀路径存在反复，利率调整仍取决于更多数据。市场开始重新评估降息节奏。",
        )
        self._insert_article(
            title="Nvidia and cloud demand keep AI capex in focus",
            category="us_technology",
            source_id="techcrunch",
            summary="AI 资本开支仍在上修，市场关注盈利兑现与估值约束。",
            content="AI 资本开支仍在上修，市场关注盈利兑现与估值约束。云计算厂商与芯片企业形成新的预期差。",
        )
        rebuild_topic_clusters(current_week_key())
        topics = self.client.get(f"/topics?week={current_week_key()}").json()

        from unittest.mock import patch

        with patch(
            "app.services.generation.refresh_market_snapshot",
            return_value=[
                {"display_name": "S&P 500", "weekly_return": 1.2, "ytd_return": 8.5},
                {"display_name": "Nasdaq 100", "weekly_return": 2.1, "ytd_return": 11.3},
            ],
        ):
            response = self.client.post(
                "/issues",
                json={
                    "week_key": current_week_key(),
                    "title": f"{current_week_key()} 推荐选题简报",
                    "topic_ids": [topic["id"] for topic in topics[:2]],
                    "generate_now": True,
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "generated")
        self.assertGreaterEqual(len(payload["drafts"]), 1)
        self.assertGreaterEqual(len(payload["selected_topics"]), 2)
        self.assertIn("title", payload["selected_topics"][0])

    def test_create_issue_download_writes_report_to_downloads(self) -> None:
        self._insert_article(
            title="Fed signals patience as inflation data stays sticky",
            category="us_economy",
            source_id="fed-press",
            summary="Fed 官员继续强调通胀路径存在反复，利率调整仍取决于更多数据。",
            content="Fed 官员继续强调通胀路径存在反复，利率调整仍取决于更多数据。市场开始重新评估降息节奏。",
        )
        self._insert_article(
            title="Taiwan expands civil defense drills as cross-strait tensions rise",
            category="cross_strait",
            source_id="bbc-zh",
            summary="台湾扩大民防演习，台海局势与两岸风险重新升温。",
            content="台湾扩大民防演习，台海局势与两岸风险重新升温。赖清德政府与大陆互动、军事部署和选举变量受到关注。",
        )
        rebuild_topic_clusters(current_week_key())
        topics = self.client.get(f"/topics?week={current_week_key()}").json()

        from unittest.mock import patch

        with patch(
            "app.services.generation.refresh_market_snapshot",
            return_value=[
                {"display_name": "S&P 500", "weekly_return": 1.2, "ytd_return": 8.5},
                {"display_name": "Nasdaq 100", "weekly_return": 2.1, "ytd_return": 11.3},
            ],
        ), patch("app.services.exports.DOWNLOAD_DIR", self._download_dir):
            response = self.client.post(
                "/ui/issues/create-download",
                data={
                    "week_key": current_week_key(),
                    "title": f"{current_week_key()} 推荐选题简报",
                    "topic_ids": [topic["id"] for topic in topics[:2]],
                    "generate_now": "true",
                    "download_format": "md",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn("attachment;", response.headers["content-disposition"])
        self.assertIn(".md", response.headers["content-disposition"])
        self.assertIn("#", response.text)

        downloaded_files = list(self._download_dir.glob("*.md"))
        self.assertEqual(len(downloaded_files), 1)
        self.assertIn("推荐选题简报", downloaded_files[0].name)
        downloaded_text = downloaded_files[0].read_text(encoding="utf-8")
        self.assertIn("相关链接：", downloaded_text)
        self.assertIn("评述：", downloaded_text)
        self.assertIn("本周总判断", downloaded_text)

    def test_recommended_topics_learn_geopolitical_focus_from_corpus(self) -> None:
        self._insert_corpus_doc(
            "每周订阅内容 20250316",
            [
                {
                    "heading": "（1）台湾反击",
                    "content": "台海局势升温，赖清德与大陆互动、军演与选举预期成为核心变量。",
                    "task_type": "topic_section",
                },
                {
                    "heading": "（2）俄乌停火谈判",
                    "content": "俄乌战争进入谈判窗口，欧洲安全架构与美国态度是更值得关注的变量。",
                    "task_type": "topic_section",
                },
            ],
        )
        self._insert_corpus_doc(
            "每周订阅内容 20251130",
            [
                {
                    "heading": "（1）美俄通话遭曝光",
                    "content": "美俄互动与乌克兰停火进程，背后是更大的地缘政治博弈。",
                    "task_type": "topic_section",
                }
            ],
        )

        self._insert_article(
            title="Taiwan expands civil defense drills as cross-strait tensions rise",
            category="cross_strait",
            source_id="bbc-zh",
            summary="台湾扩大民防演习，台海局势与两岸风险重新升温。",
            content="台湾扩大民防演习，台海局势与两岸风险重新升温。赖清德政府与大陆互动、军事部署和选举变量受到关注。",
        )
        self._insert_article(
            title="Ukraine ceasefire talks reopen questions about NATO guarantees",
            category="russia_ukraine",
            source_id="bbc-zh",
            summary="俄乌停火谈判再次启动，欧洲安全与北约承诺成为焦点。",
            content="俄乌停火谈判再次启动，欧洲安全与北约承诺成为焦点。乌克兰、俄罗斯与美国之间的谈判边界值得关注。",
        )
        self._insert_article(
            title="New AI coding assistant targets enterprise workflows",
            category="us_technology",
            source_id="techcrunch",
            summary="A startup launches a new AI coding assistant for enterprise workflows.",
            content="A startup launches a new AI coding assistant for enterprise workflows, highlighting model orchestration and developer productivity.",
        )

        rebuild_topic_clusters(current_week_key())
        recommended = recommend_topics(current_week_key(), min_count=2, max_count=3)
        top_titles = [topic["title"] for topic in recommended[:2]]

        self.assertIn("Taiwan expands civil defense drills as cross-strait tensions rise", top_titles)
        self.assertIn("Ukraine ceasefire talks reopen questions about NATO guarantees", top_titles)
        self.assertNotEqual(recommended[0]["category"], "us_technology")

    def test_topic_preference_overview_learns_priority_from_corpus_topics(self) -> None:
        self._insert_corpus_doc(
            "每周订阅内容 20250316",
            [
                {
                    "heading": "（1）台湾反击",
                    "content": "台海局势升温，赖清德与大陆互动、军演与选举预期成为核心变量。",
                    "task_type": "topic_section",
                },
                {
                    "heading": "（2）俄乌停火谈判",
                    "content": "俄乌战争进入谈判窗口，欧洲安全架构与美国态度是更值得关注的变量。",
                    "task_type": "topic_section",
                },
            ],
        )
        self._insert_corpus_doc(
            "每周订阅内容 20260308",
            [
                {
                    "heading": "（1）伊朗局势",
                    "content": "中东局势升级，伊朗和以色列冲突对全球风险偏好有重要影响。",
                    "task_type": "topic_section",
                }
            ],
        )

        overview = topic_preference_overview(limit_docs=10)
        labels = [item["label"] for item in overview["priority_categories"]]

        self.assertIn("台海", labels)
        self.assertIn("俄乌", labels)
        self.assertIn("中东", labels)
        self.assertIn("台湾", overview["priority_keywords"])

    def test_create_issue_records_topic_selection_feedback_pairs(self) -> None:
        self._insert_corpus_doc(
            "每周订阅内容 20250316",
            [
                {
                    "heading": "（1）台湾反击",
                    "content": "台海局势升温，赖清德与大陆互动、军演与选举预期成为核心变量。",
                    "task_type": "topic_section",
                },
                {
                    "heading": "（2）俄乌停火谈判",
                    "content": "俄乌停火谈判、美国承诺与欧洲安全架构是更重要的长期变量。",
                    "task_type": "topic_section",
                },
            ],
        )
        self._insert_article(
            title="Taiwan expands civil defense drills as cross-strait tensions rise",
            category="cross_strait",
            source_id="bbc-zh",
            summary="台湾扩大民防演习，台海局势与两岸风险重新升温。",
            content="台湾扩大民防演习，台海局势与两岸风险重新升温。赖清德政府与大陆互动、军事部署和选举变量受到关注。",
        )
        self._insert_article(
            title="Ukraine ceasefire talks reopen questions about NATO guarantees",
            category="russia_ukraine",
            source_id="bbc-zh",
            summary="俄乌停火谈判再次启动，欧洲安全与北约承诺成为焦点。",
            content="俄乌停火谈判再次启动，欧洲安全与北约承诺成为焦点。乌克兰、俄罗斯与美国之间的谈判边界值得关注。",
        )
        self._insert_article(
            title="New AI coding assistant targets enterprise workflows",
            category="us_technology",
            source_id="techcrunch",
            summary="A startup launches a new AI coding assistant for enterprise workflows.",
            content="A startup launches a new AI coding assistant for enterprise workflows, highlighting model orchestration and developer productivity.",
        )

        rebuild_topic_clusters(current_week_key())
        recommended = recommend_topics(current_week_key(), min_count=2, max_count=3)
        response = self.client.post(
            "/issues",
            json={
                "week_key": current_week_key(),
                "title": f"{current_week_key()} 推荐选题简报",
                "topic_ids": [recommended[0]["id"]],
                "generate_now": False,
            },
        )

        self.assertEqual(response.status_code, 200)
        with get_connection() as connection:
            rows = connection.execute(
                """
                SELECT chosen_text, rejected_text, reason
                FROM preference_pairs
                WHERE reason = 'topic_selection_feedback'
                """
            ).fetchall()

        self.assertGreaterEqual(len(rows), 1)
        self.assertIn(recommended[0]["title"], rows[0]["chosen_text"])
        self.assertTrue(any(row["rejected_text"] for row in rows))

        overview = topic_preference_overview(limit_docs=10)
        self.assertGreaterEqual(overview["selection_feedback_count"], 1)

        preference_response = self.client.post("/training/preferences")
        self.assertEqual(preference_response.status_code, 200)
        self.assertGreaterEqual(preference_response.json()["metrics"]["selection_feedback_pair_count"], 1)

    def test_rebuild_topic_clusters_demotes_weak_consumer_topics(self) -> None:
        self._insert_article(
            title="Where to buy the new iPhone 17E",
            category="us_technology",
            source_id="the-verge",
            summary="A preorder and price guide covering where shoppers can buy the new iPhone 17E.",
            content="This guide explains where to buy the new iPhone 17E, preorder timing, price tiers, and carrier deals.",
        )
        self._insert_article(
            title="Google wraps up $32B acquisition of cloud cybersecurity startup Wiz",
            category="us_technology",
            source_id="techcrunch",
            summary="Google completes its Wiz deal to deepen cloud security and enterprise software reach.",
            content="Google completed the Wiz acquisition, reinforcing cloud security, enterprise software positioning, and AI infrastructure ambitions.",
        )

        clusters = rebuild_topic_clusters(current_week_key())
        titles = [cluster["title"] for cluster in clusters]
        self.assertIn("Google wraps up $32B acquisition of cloud cybersecurity startup Wiz", titles)
        self.assertEqual(titles[0], "Google wraps up $32B acquisition of cloud cybersecurity startup Wiz")

        weak_cluster = next(cluster for cluster in clusters if cluster["title"] == "Where to buy the new iPhone 17E")
        self.assertIn("weak_topic", weak_cluster["evidence_pack"]["editorial_flags"])
        self.assertLess(weak_cluster["score"], clusters[0]["score"])
        self.assertTrue(weak_cluster["evidence_pack"]["impact_paths"])
        self.assertTrue(weak_cluster["evidence_pack"]["confidence_note"])

    def test_runtime_can_force_openai_when_available(self) -> None:
        fake_client = Mock()
        fake_client.available.return_value = True
        fake_client.model = "gpt-5.4"
        with patch("app.services.runtime.OpenAIEditorialClient", return_value=fake_client):
            runtime = LocalWriterRuntime(force_backend="openai")
        self.assertEqual(runtime.describe()["backend"], "openai")
        self.assertEqual(runtime.describe()["base_model"], "gpt-5.4")

    def test_runtime_stays_local_by_default_even_if_openai_is_available(self) -> None:
        fake_client = Mock()
        fake_client.available.return_value = True
        fake_client.model = "gpt-5.4"
        with patch("app.services.runtime.OpenAIEditorialClient", return_value=fake_client):
            runtime = LocalWriterRuntime()
        self.assertNotEqual(runtime.describe()["backend"], "openai")

    def test_runtime_openai_candidate_generation_uses_structured_brief(self) -> None:
        fake_client = Mock()
        fake_client.available.return_value = True
        fake_client.model = "gpt-5.4"
        fake_client.create_json.side_effect = [
            {
                "angle": "先确认政策动作，再判断它对风险偏好的改变。",
                "fact_chain": ["美国宣布新一轮限制措施", "相关部门给出执行时间表"],
                "why_it_matters": ["这改变了政策边界", "也会改变企业与市场的预期"],
                "judgement": "我的判断是，市场还没有把持续性风险完全计入。",
                "market_link": "科技估值与风险溢价会先反应。",
                "risk_watch": ["后续执行细则", "盟友是否跟进"],
                "confidence_boundary": "当前可以做方向判断，但还不能写成定局。",
            }
        ]
        fake_client.create_text.side_effect = [
            "美国宣布新一轮限制措施。更重要的是，市场还没有把政策持续性完全计入。",
            "把时间线拉长看，这不是一次性噪音。我的判断是，风险溢价还会继续抬高。",
            "相关部门给出执行时间表后，真正关键的是企业行为和估值框架会不会跟着变。",
        ]
        with patch("app.services.runtime.OpenAIEditorialClient", return_value=fake_client):
            runtime = LocalWriterRuntime(force_backend="openai")

        candidates = runtime.generate_candidates(
            "美国限制升级",
            {
                "event_summary": "美国宣布新一轮限制措施，并公布执行时间表。",
                "key_facts": ["美国宣布新一轮限制措施", "相关部门给出执行时间表"],
                "entities": ["美国", "相关部门"],
                "official_sources": ["https://example.com/official"],
                "cross_sources": ["https://example.com/media"],
                "impact_paths": ["政策边界变化会先传导到风险溢价。"],
                "allowed_conclusions": ["可以讨论政策边界与市场预期变化。"],
                "forbidden_claims": ["不要把未落地政策写成定局。"],
                "confidence_note": "当前可以做方向判断，但还不能写成定局。",
            },
            {
                "voice_notes": ["判断要鲜明，但不要喊口号。"],
                "worldview_notes": ["重视政策边界与资产定价。"],
                "banned_phrases": [],
                "preferred_patterns": ["主题标题 -> 相关链接 -> 评述"],
            },
            market_snapshot=[{"display_name": "S&P 500", "weekly_return": -1.2, "ytd_return": 3.4, "latest_close": 5100}],
            category="us_politics",
        )

        self.assertEqual(len(candidates), 3)
        self.assertTrue(all(candidate["sampling_note"].startswith("openai_candidate_") for candidate in candidates))
        self.assertTrue(any("我的判断是" in candidate["text"] or "更重要的是" in candidate["text"] for candidate in candidates))

    def test_candidate_scoring_penalizes_off_topic_leakage_and_dedupes_citations(self) -> None:
        evidence_pack = {
            "event_summary": "Google completed its $32B acquisition of Wiz.",
            "key_facts": [
                "Google wraps up $32B acquisition of cloud cybersecurity startup Wiz",
                "The deal deepens Google's cloud security offering.",
            ],
            "entities": ["Google", "Wiz", "cloud security"],
            "official_sources": ["https://example.com/google-wiz", "https://example.com/google-wiz"],
            "cross_sources": ["https://example.com/google-wiz", "https://example.com/analysis"],
        }
        style_profile = {"banned_phrases": []}
        clean_candidate = (
            "Google wraps up $32B acquisition of cloud cybersecurity startup Wiz，这笔交易更值得关注的地方，"
            "是Google试图把云安全变成云业务增长的一部分。"
        )
        leaked_candidate = clean_candidate + " 与此同时，房地产按揭和LPR下行也会带动写字楼去库存。"

        clean_review = score_candidate(clean_candidate, evidence_pack, style_profile)
        leaked_review = score_candidate(leaked_candidate, evidence_pack, style_profile)

        self.assertGreater(clean_review["score"], leaked_review["score"])
        self.assertTrue(any("疑似串文污染" in note for note in leaked_review["notes"]))

        citations = build_citation_bundle(evidence_pack)
        self.assertEqual(
            citations,
            [
                {"index": 1, "url": "https://example.com/google-wiz"},
                {"index": 2, "url": "https://example.com/analysis"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
