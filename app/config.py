from __future__ import annotations

import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
EXPORT_DIR = DATA_DIR / "exports"
TRAINING_DIR = DATA_DIR / "training"
MODEL_DIR = DATA_DIR / "models"
DB_PATH = DATA_DIR / "studio.sqlite3"
DOWNLOAD_DIR = Path.home() / "Downloads"

APP_TITLE = "Local Briefing Studio"
DEFAULT_WEEKLY_TARGET_TOPICS = 5
TARGET_LENGTH_RANGE = (4500, 6500)


def _read_secret(env_name: str, fallback_path: Path) -> str:
    direct_value = (os.getenv(env_name) or "").strip()
    if direct_value:
        return direct_value
    try:
        return fallback_path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


OPENAI_API_KEY = _read_secret("OPENAI_API_KEY", DATA_DIR / "openai_api_key.txt")
OPENAI_BASE_URL = (os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
OPENAI_MODEL = (os.getenv("OPENAI_MODEL") or "gpt-5.4").strip() or "gpt-5.4"
OPENAI_REASONING_EFFORT = (os.getenv("OPENAI_REASONING_EFFORT") or "medium").strip() or "medium"
OPENAI_TIMEOUT_SECONDS = float(os.getenv("OPENAI_TIMEOUT_SECONDS") or "45")
WRITER_BACKEND_POLICY = (os.getenv("LBS_WRITER_BACKEND") or "local_first").strip() or "local_first"
TOPIC_REFINEMENT_LIMIT = max(0, int(os.getenv("LBS_TOPIC_REFINEMENT_LIMIT") or "15"))

DEFAULT_MARKET_SYMBOLS = {
    "^GSPC": {"label": "S&P 500", "kind": "index"},
    "^NDX": {"label": "Nasdaq 100", "kind": "index"},
    "^VIX": {"label": "VIX", "kind": "index"},
    "^TNX": {"label": "US 10Y", "kind": "macro"},
    "DX-Y.NYB": {"label": "DXY", "kind": "macro"},
    "AAPL": {"label": "Apple", "kind": "equity"},
    "MSFT": {"label": "Microsoft", "kind": "equity"},
    "NVDA": {"label": "NVIDIA", "kind": "equity"},
    "AMZN": {"label": "Amazon", "kind": "equity"},
    "GOOG": {"label": "Alphabet", "kind": "equity"},
    "META": {"label": "Meta", "kind": "equity"},
    "TSLA": {"label": "Tesla", "kind": "equity"},
    "AVGO": {"label": "Broadcom", "kind": "equity"},
    "AMD": {"label": "AMD", "kind": "equity"},
    "TSM": {"label": "TSMC", "kind": "equity"},
}

DEFAULT_SOURCE_SEEDS = [
    {
        "id": "fed-press",
        "name": "Federal Reserve Press Releases",
        "base_url": "https://www.federalreserve.gov",
        "feed_url": "https://www.federalreserve.gov/feeds/press_all.xml",
        "language": "en",
        "source_type": "official",
        "categories": ["us_economy", "us_politics"],
    },
    {
        "id": "sec-press",
        "name": "SEC Press Releases",
        "base_url": "https://www.sec.gov",
        "feed_url": "https://www.sec.gov/news/pressreleases.rss",
        "language": "en",
        "source_type": "official",
        "categories": ["us_politics", "us_technology"],
    },
    {
        "id": "techcrunch",
        "name": "TechCrunch",
        "base_url": "https://techcrunch.com",
        "feed_url": "https://techcrunch.com/feed/",
        "language": "en",
        "source_type": "media",
        "categories": ["us_technology"],
    },
    {
        "id": "the-verge",
        "name": "The Verge",
        "base_url": "https://www.theverge.com",
        "feed_url": "https://www.theverge.com/rss/index.xml",
        "language": "en",
        "source_type": "media",
        "categories": ["us_technology"],
    },
    {
        "id": "white-house",
        "name": "The White House Briefing Room",
        "base_url": "https://www.whitehouse.gov",
        "feed_url": "",
        "page_url": "https://www.whitehouse.gov/briefing-room/",
        "language": "en",
        "source_type": "official",
        "categories": ["us_politics"],
    },
    {
        "id": "bbc-zh",
        "name": "BBC News 中文",
        "base_url": "https://www.bbc.com",
        "feed_url": "",
        "page_url": "https://www.bbc.com/zhongwen/simp",
        "language": "zh",
        "source_type": "media",
        "categories": ["cross_strait", "russia_ukraine", "middle_east", "global_geopolitics", "china_politics", "us_politics"],
    },
    {
        "id": "voa-zh",
        "name": "VOA 中文",
        "base_url": "https://www.voachinese.com",
        "feed_url": "",
        "page_url": "https://www.voachinese.com/",
        "language": "zh",
        "source_type": "media",
        "categories": ["cross_strait", "russia_ukraine", "middle_east", "global_geopolitics", "china_politics", "us_politics"],
    },
    {
        "id": "gov-cn",
        "name": "中国政府网要闻",
        "base_url": "https://www.gov.cn",
        "feed_url": "",
        "page_url": "https://www.gov.cn/yaowen/",
        "language": "zh",
        "source_type": "official",
        "categories": ["china_macro", "china_politics"],
    },
]


def ensure_directories() -> None:
    for path in [DATA_DIR, UPLOAD_DIR, EXPORT_DIR, TRAINING_DIR, MODEL_DIR, DOWNLOAD_DIR]:
        path.mkdir(parents=True, exist_ok=True)
