from __future__ import annotations

import uuid
from datetime import date, timedelta
from typing import Any, Optional

import yfinance as yf

from ..config import DEFAULT_MARKET_SYMBOLS
from ..db import get_connection
from ..utils import current_week_key, json_dumps, json_loads, now_iso


def refresh_market_snapshot(week_key: Optional[str] = None) -> list[dict[str, Any]]:
    week_key = week_key or current_week_key()
    start = date.today() - timedelta(days=14)
    end = date.today()
    results: list[dict[str, Any]] = []

    for symbol, meta in DEFAULT_MARKET_SYMBOLS.items():
        try:
            ticker = yf.Ticker(symbol)
            history = ticker.history(start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(), interval="1d")
            if history.empty:
                continue
            closes = [round(float(value), 4) for value in history["Close"].tolist()]
            dates = [index.strftime("%Y-%m-%d") for index in history.index.to_pydatetime()]
            latest_close = closes[-1]
            weekly_return = ((closes[-1] / closes[max(0, len(closes) - 6)]) - 1) * 100 if len(closes) > 5 else 0.0
            ytd_history = ticker.history(start=date(end.year, 1, 1).isoformat(), end=(end + timedelta(days=1)).isoformat(), interval="1d")
            ytd_closes = [float(value) for value in ytd_history["Close"].tolist()] if not ytd_history.empty else closes
            ytd_return = ((ytd_closes[-1] / ytd_closes[0]) - 1) * 100 if len(ytd_closes) > 1 else 0.0
            record = {
                "id": uuid.uuid4().hex,
                "week_key": week_key,
                "symbol": symbol,
                "display_name": meta["label"],
                "asset_type": meta["kind"],
                "window_start": dates[0],
                "window_end": dates[-1],
                "latest_close": latest_close,
                "weekly_return": round(weekly_return, 2),
                "ytd_return": round(ytd_return, 2),
                "notes": f"{meta['label']} 在最近一周内波动 {round(max(closes) - min(closes), 2)}。",
                "raw_series": [{"date": item_date, "close": item_close} for item_date, item_close in zip(dates, closes)],
            }
            results.append(record)
        except Exception:
            continue

    with get_connection() as connection:
        for record in results:
            connection.execute(
                """
                INSERT OR REPLACE INTO market_snapshots (
                  id, week_key, symbol, display_name, asset_type, window_start, window_end,
                  latest_close, weekly_return, ytd_return, notes, related_article_urls_json,
                  raw_series_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["id"],
                    record["week_key"],
                    record["symbol"],
                    record["display_name"],
                    record["asset_type"],
                    record["window_start"],
                    record["window_end"],
                    record["latest_close"],
                    record["weekly_return"],
                    record["ytd_return"],
                    record["notes"],
                    json_dumps([]),
                    json_dumps(record["raw_series"]),
                    now_iso(),
                ),
            )
    return results


def latest_market_snapshot(week_key: Optional[str] = None) -> list[dict[str, Any]]:
    week_key = week_key or current_week_key()
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT symbol, display_name, asset_type, latest_close, weekly_return, ytd_return, notes, raw_series_json
            FROM market_snapshots WHERE week_key = ? ORDER BY asset_type, symbol
            """,
            (week_key,),
        ).fetchall()
    return [
        {
            "symbol": row["symbol"],
            "display_name": row["display_name"],
            "asset_type": row["asset_type"],
            "latest_close": row["latest_close"],
            "weekly_return": row["weekly_return"],
            "ytd_return": row["ytd_return"],
            "notes": row["notes"],
            "raw_series": json_loads(row["raw_series_json"], []),
        }
        for row in rows
    ]
