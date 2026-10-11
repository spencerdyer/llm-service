"""Metrics collection and storage service."""

import asyncio
import json
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

import aiosqlite

from llm_service.config import settings


def clip_log_text(value: Any, limit: Optional[int] = None) -> Optional[str]:
    """Store a text body, clipping to ``request_log_max_chars`` if needed."""
    if value is None:
        return None
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, default=str)
    if not value:
        return value
    max_chars = limit if limit is not None else int(settings.request_log_max_chars or 1_000_000)
    max_chars = max(10_000, max_chars)
    if len(value) <= max_chars:
        return value
    omitted = len(value) - max_chars
    return value[:max_chars] + f"\n\n… [truncated {omitted:,} characters for storage]"


def explain_generation_end(
    *,
    status: str,
    finish_reason: Optional[str] = None,
    engine_finish_reason: Optional[str] = None,
    max_tokens: Optional[int] = None,
    completion_tokens: int = 0,
    token_count_source: str = "estimated",
    stop: Optional[list] = None,
    error: Optional[str] = None,
    reasoning_chars: int = 0,
    answer_chars: int = 0,
) -> tuple[str, bool]:
    """Human-readable reason the request ended, plus a truncated flag."""
    status = (status or "ok").lower()
    client_reason = (finish_reason or "").strip().lower()
    engine_reason = (engine_finish_reason or "").strip().lower()
    hit_length = engine_reason == "length" or client_reason == "length"
    at_cap = bool(max_tokens) and completion_tokens >= int(max_tokens)
    truncated = status in {"cancelled", "error"} or hit_length or at_cap or engine_reason in {
        "abort",
        "timeout",
    }

    count_note = (
        f"{completion_tokens} completion tokens"
        + (f" of max_tokens={max_tokens}" if max_tokens else "")
        + ("" if token_count_source == "engine" else " (character-estimated; not engine counts)")
    )

    if status == "cancelled":
        return (
            "Client disconnected or the request was aborted before generation finished. "
            f"Partial output saved ({count_note}).",
            True,
        )
    if status == "error":
        return (f"Generation failed: {error or 'unknown error'}. {count_note}.", True)
    if engine_reason == "timeout" or client_reason == "timeout":
        return (f"Backend stream timed out. Partial output saved ({count_note}).", True)
    if engine_reason == "abort" or client_reason == "abort":
        return (f"The engine aborted the request. Partial output saved ({count_note}).", True)
    if hit_length or at_cap:
        think_note = ""
        if reasoning_chars and reasoning_chars >= max(answer_chars, 1) * 2:
            think_note = (
                " Most of the output is inside <think>…</think>, so the visible answer "
                "can look unfinished even though the token budget was spent."
            )
        return (
            f"Hit the max token cap ({count_note}). The response is truncated; raise "
            "max_completion_tokens (Grok) or the model's max_tokens / tool_max_tokens."
            f"{think_note}",
            True,
        )
    if client_reason == "tool_calls":
        extra = ""
        if engine_reason == "length":
            extra = " Engine also reported length, so the tool call may be incomplete."
        return (f"Model emitted a tool call and stopped for the tool loop. {count_note}.{extra}", truncated)
    if client_reason == "stop" or engine_reason == "stop":
        stops = ""
        if stop:
            stops = f" Configured stop sequences: {stop}."
        return (
            f"Model emitted an end-of-sequence token (natural stop). {count_note}.{stops}",
            False,
        )
    if client_reason.startswith("error") or engine_reason.startswith("error"):
        return (f"Engine error ({finish_reason or engine_finish_reason}). {count_note}.", True)
    if not finish_reason and not engine_finish_reason:
        return (
            f"Stream ended without a finish_reason. Output may be partial. {count_note}.",
            True,
        )
    return (
        f"Finished with finish_reason={finish_reason or 'none'} "
        f"(engine={engine_finish_reason or 'none'}). {count_note}.",
        truncated,
    )


@dataclass
class MetricPoint:
    """A single metric data point."""
    timestamp: float
    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Tokens per second stats
    tps_min: float = 0.0
    tps_max: float = 0.0
    tps_sum: float = 0.0
    tps_count: int = 0

    @property
    def tps_avg(self) -> float:
        return self.tps_sum / self.tps_count if self.tps_count > 0 else 0.0


class MetricsService:
    """Collects and stores API metrics."""

    def __init__(self):
        self.db_path = settings.effective_db_path
        self._db: Optional[aiosqlite.Connection] = None

        # In-memory buffer for recent metrics (last 60 seconds, per-second granularity)
        self._realtime_buffer: deque[MetricPoint] = deque(maxlen=60)

        # Current second accumulator
        self._current_second: int = 0
        self._current_requests: int = 0
        self._current_prompt_tokens: int = 0
        self._current_completion_tokens: int = 0
        # Tokens per second tracking for current second
        self._current_tps_min: float = float('inf')
        self._current_tps_max: float = 0.0
        self._current_tps_sum: float = 0.0
        self._current_tps_count: int = 0

        # Lock for thread safety
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize the metrics service."""
        self._db = await aiosqlite.connect(self.db_path)
        await self._create_tables()
        self._current_second = int(time.time())

    async def _create_tables(self) -> None:
        """Create metrics tables."""
        # Per-minute aggregated metrics for historical data
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS metrics_minute (
                timestamp INTEGER PRIMARY KEY,
                requests INTEGER DEFAULT 0,
                prompt_tokens INTEGER DEFAULT 0,
                completion_tokens INTEGER DEFAULT 0,
                tps_min REAL DEFAULT 0,
                tps_max REAL DEFAULT 0,
                tps_sum REAL DEFAULT 0,
                tps_count INTEGER DEFAULT 0
            )
        """)

        # Add tps columns if they don't exist (migration for existing DBs)
        try:
            await self._db.execute("ALTER TABLE metrics_minute ADD COLUMN tps_min REAL DEFAULT 0")
        except Exception:
            pass  # Column already exists
        try:
            await self._db.execute("ALTER TABLE metrics_minute ADD COLUMN tps_max REAL DEFAULT 0")
        except Exception:
            pass
        try:
            await self._db.execute("ALTER TABLE metrics_minute ADD COLUMN tps_sum REAL DEFAULT 0")
        except Exception:
            pass
        try:
            await self._db.execute("ALTER TABLE metrics_minute ADD COLUMN tps_count INTEGER DEFAULT 0")
        except Exception:
            pass

        # Index for time-based queries
        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_metrics_timestamp
            ON metrics_minute(timestamp)
        """)

        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS request_logs (
                id TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                model TEXT,
                endpoint TEXT,
                stream INTEGER DEFAULT 0,
                status TEXT,
                prompt_tokens INTEGER DEFAULT 0,
                completion_tokens INTEGER DEFAULT 0,
                duration_ms REAL,
                tokens_per_second REAL,
                finish_reason TEXT,
                has_tools INTEGER DEFAULT 0,
                tool_choice TEXT,
                tool_names TEXT,
                rejected_tool_names TEXT,
                prompt_preview TEXT,
                completion_preview TEXT,
                error TEXT
            )
        """)
        for column, ddl in (
            ("prompt_text", "TEXT"),
            ("completion_text", "TEXT"),
            ("reasoning_text", "TEXT"),
            ("messages_json", "TEXT"),
            ("sampling_json", "TEXT"),
            ("stop_explanation", "TEXT"),
            ("max_tokens", "INTEGER"),
            ("engine_finish_reason", "TEXT"),
            ("token_count_source", "TEXT"),
            ("truncated", "INTEGER DEFAULT 0"),
            ("tool_calls_json", "TEXT"),
        ):
            try:
                await self._db.execute(
                    f"ALTER TABLE request_logs ADD COLUMN {column} {ddl}"
                )
            except Exception:
                pass
        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_request_logs_timestamp
            ON request_logs(timestamp DESC)
        """)

        await self._db.commit()

    async def record_request(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        tokens_per_second: float = 0.0,
        **log: Any,
    ) -> None:
        """Record a completed request with token counts and generation speed."""
        async with self._lock:
            now = int(time.time())

            # If we've moved to a new second, flush the previous one
            if now != self._current_second:
                await self._flush_current_second()
                self._current_second = now

            # Accumulate in current second
            self._current_requests += 1
            self._current_prompt_tokens += prompt_tokens
            self._current_completion_tokens += completion_tokens

            # Track tokens per second stats
            if tokens_per_second > 0:
                self._current_tps_min = min(self._current_tps_min, tokens_per_second)
                self._current_tps_max = max(self._current_tps_max, tokens_per_second)
                self._current_tps_sum += tokens_per_second
                self._current_tps_count += 1

            if log:
                await self._write_call_log(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    tokens_per_second=tokens_per_second,
                    log=log,
                )

    async def _write_call_log(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        tokens_per_second: float,
        log: dict[str, Any],
    ) -> None:
        request_id = log.get("request_id") or f"req-{int(time.time() * 1000)}"
        tool_names = log.get("tool_names") or []
        rejected = log.get("rejected_tool_names") or []
        tool_calls = log.get("tool_calls")
        sampling = log.get("sampling")
        messages = log.get("messages")
        truncated = 1 if log.get("truncated") else 0
        await self._db.execute(
            """
            INSERT OR REPLACE INTO request_logs (
                id, timestamp, model, endpoint, stream, status,
                prompt_tokens, completion_tokens, duration_ms, tokens_per_second,
                finish_reason, has_tools, tool_choice, tool_names, rejected_tool_names,
                prompt_preview, completion_preview, error,
                prompt_text, completion_text, reasoning_text, messages_json, sampling_json,
                stop_explanation, max_tokens, engine_finish_reason, token_count_source,
                truncated, tool_calls_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                request_id,
                log.get("timestamp") or time.time(),
                log.get("model"),
                log.get("endpoint") or "chat.completions",
                1 if log.get("stream") else 0,
                log.get("status") or "ok",
                prompt_tokens,
                completion_tokens,
                log.get("duration_ms"),
                round(tokens_per_second, 2) if tokens_per_second else 0,
                log.get("finish_reason"),
                1 if log.get("has_tools") else 0,
                log.get("tool_choice"),
                json.dumps(tool_names) if tool_names else None,
                json.dumps(rejected) if rejected else None,
                log.get("prompt_preview"),
                log.get("completion_preview"),
                log.get("error"),
                clip_log_text(log.get("prompt_text")),
                clip_log_text(log.get("completion_text")),
                clip_log_text(log.get("reasoning_text")),
                clip_log_text(
                    json.dumps(messages, ensure_ascii=False, default=str) if messages else None
                ),
                json.dumps(sampling, ensure_ascii=False, default=str) if sampling else None,
                log.get("stop_explanation"),
                log.get("max_tokens"),
                log.get("engine_finish_reason"),
                log.get("token_count_source"),
                truncated,
                json.dumps(tool_calls, ensure_ascii=False, default=str) if tool_calls else None,
            ),
        )
        await self._db.commit()
        await self._trim_call_logs()

    async def _trim_call_logs(self) -> None:
        max_rows = max(100, int(settings.request_log_max_rows or 5000))
        async with self._db.execute("SELECT COUNT(*) FROM request_logs") as cursor:
            count = (await cursor.fetchone())[0]
        if count <= max_rows:
            return
        overflow = count - max_rows
        async with self._db.execute(
            "SELECT id FROM request_logs ORDER BY timestamp ASC LIMIT ?",
            (overflow,),
        ) as cursor:
            drop = [row[0] for row in await cursor.fetchall()]
        if not drop:
            return
        placeholders = ",".join("?" * len(drop))
        await self._db.execute(
            f"DELETE FROM request_logs WHERE id IN ({placeholders})",
            drop,
        )
        await self._db.commit()

    _LIST_COLUMNS = (
        "id",
        "timestamp",
        "model",
        "endpoint",
        "stream",
        "status",
        "prompt_tokens",
        "completion_tokens",
        "duration_ms",
        "tokens_per_second",
        "finish_reason",
        "has_tools",
        "tool_choice",
        "tool_names",
        "rejected_tool_names",
        "prompt_preview",
        "completion_preview",
        "error",
        "stop_explanation",
        "max_tokens",
        "engine_finish_reason",
        "token_count_source",
        "truncated",
    )
    _DETAIL_COLUMNS = _LIST_COLUMNS + (
        "prompt_text",
        "completion_text",
        "reasoning_text",
        "messages_json",
        "sampling_json",
        "tool_calls_json",
    )

    @classmethod
    def _row_to_call(cls, row: tuple, columns: Optional[tuple[str, ...]] = None) -> dict:
        def _loads_list(value):
            if not value:
                return []
            try:
                parsed = json.loads(value)
                return parsed if isinstance(parsed, list) else []
            except json.JSONDecodeError:
                return []

        def _loads_any(value):
            if not value:
                return None
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value

        cols = columns or cls._DETAIL_COLUMNS
        data = dict(zip(cols, row))
        call = {
            "id": data.get("id"),
            "timestamp": data.get("timestamp"),
            "model": data.get("model"),
            "endpoint": data.get("endpoint"),
            "stream": bool(data.get("stream")),
            "status": data.get("status"),
            "prompt_tokens": data.get("prompt_tokens") or 0,
            "completion_tokens": data.get("completion_tokens") or 0,
            "duration_ms": data.get("duration_ms"),
            "tokens_per_second": data.get("tokens_per_second"),
            "finish_reason": data.get("finish_reason"),
            "has_tools": bool(data.get("has_tools")),
            "tool_choice": data.get("tool_choice"),
            "tool_names": _loads_list(data.get("tool_names")),
            "rejected_tool_names": _loads_list(data.get("rejected_tool_names")),
            "prompt_preview": data.get("prompt_preview"),
            "completion_preview": data.get("completion_preview"),
            "error": data.get("error"),
            "stop_explanation": data.get("stop_explanation"),
            "max_tokens": data.get("max_tokens"),
            "engine_finish_reason": data.get("engine_finish_reason"),
            "token_count_source": data.get("token_count_source"),
            "truncated": bool(data.get("truncated")),
        }
        if "prompt_text" in data:
            call["prompt_text"] = data.get("prompt_text")
            call["completion_text"] = data.get("completion_text")
            call["reasoning_text"] = data.get("reasoning_text")
            call["messages"] = _loads_any(data.get("messages_json"))
            call["sampling"] = _loads_any(data.get("sampling_json"))
            call["tool_calls"] = _loads_any(data.get("tool_calls_json"))
        return call

    async def list_call_logs(
        self,
        limit: int = 50,
        offset: int = 0,
        model: Optional[str] = None,
        status: Optional[str] = None,
        has_tools: Optional[bool] = None,
        q: Optional[str] = None,
    ) -> dict:
        """Return recent API call logs with optional filters."""
        limit = max(1, min(limit, 200))
        offset = max(0, offset)
        clauses = ["1=1"]
        params: list[Any] = []
        if model:
            clauses.append("model = ?")
            params.append(model)
        if status:
            clauses.append("status = ?")
            params.append(status)
        if has_tools is not None:
            clauses.append("has_tools = ?")
            params.append(1 if has_tools else 0)
        if q:
            clauses.append(
                "(prompt_preview LIKE ? OR completion_preview LIKE ? OR prompt_text LIKE ? "
                "OR completion_text LIKE ? OR stop_explanation LIKE ? OR tool_names LIKE ? "
                "OR rejected_tool_names LIKE ? OR error LIKE ?)"
            )
            like = f"%{q}%"
            params.extend([like] * 8)
        where = " AND ".join(clauses)

        async with self._db.execute(
            f"SELECT COUNT(*) FROM request_logs WHERE {where}", params
        ) as cursor:
            total = (await cursor.fetchone())[0]

        columns = ", ".join(self._LIST_COLUMNS)
        async with self._db.execute(
            f"""
            SELECT {columns} FROM request_logs
            WHERE {where}
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
            """,
            [*params, limit, offset],
        ) as cursor:
            rows = await cursor.fetchall()
        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "calls": [self._row_to_call(row, self._LIST_COLUMNS) for row in rows],
        }

    async def get_call_log(self, call_id: str) -> Optional[dict]:
        columns = ", ".join(self._DETAIL_COLUMNS)
        async with self._db.execute(
            f"SELECT {columns} FROM request_logs WHERE id = ?", (call_id,)
        ) as cursor:
            row = await cursor.fetchone()
        if not row:
            return None
        return self._row_to_call(row, self._DETAIL_COLUMNS)

    async def clear_call_logs(self) -> None:
        await self._db.execute("DELETE FROM request_logs")
        await self._db.commit()

    async def _flush_current_second(self) -> None:
        """Flush current second data to buffer and potentially to DB."""
        if self._current_requests > 0 or self._current_prompt_tokens > 0:
            # Handle case where no valid TPS measurements
            tps_min = self._current_tps_min if self._current_tps_count > 0 else 0.0
            tps_max = self._current_tps_max if self._current_tps_count > 0 else 0.0

            point = MetricPoint(
                timestamp=self._current_second,
                requests=self._current_requests,
                prompt_tokens=self._current_prompt_tokens,
                completion_tokens=self._current_completion_tokens,
                tps_min=tps_min,
                tps_max=tps_max,
                tps_sum=self._current_tps_sum,
                tps_count=self._current_tps_count,
            )
            self._realtime_buffer.append(point)

            # Aggregate to per-minute storage
            minute_ts = (self._current_second // 60) * 60
            await self._db.execute("""
                INSERT INTO metrics_minute (timestamp, requests, prompt_tokens, completion_tokens, tps_min, tps_max, tps_sum, tps_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(timestamp) DO UPDATE SET
                    requests = requests + excluded.requests,
                    prompt_tokens = prompt_tokens + excluded.prompt_tokens,
                    completion_tokens = completion_tokens + excluded.completion_tokens,
                    tps_min = CASE WHEN excluded.tps_min > 0 AND (metrics_minute.tps_min = 0 OR excluded.tps_min < metrics_minute.tps_min) THEN excluded.tps_min ELSE metrics_minute.tps_min END,
                    tps_max = CASE WHEN excluded.tps_max > metrics_minute.tps_max THEN excluded.tps_max ELSE metrics_minute.tps_max END,
                    tps_sum = tps_sum + excluded.tps_sum,
                    tps_count = tps_count + excluded.tps_count
            """, (minute_ts, self._current_requests, self._current_prompt_tokens, self._current_completion_tokens,
                  tps_min, tps_max, self._current_tps_sum, self._current_tps_count))
            await self._db.commit()

        # Reset accumulators
        self._current_requests = 0
        self._current_prompt_tokens = 0
        self._current_completion_tokens = 0
        self._current_tps_min = float('inf')
        self._current_tps_max = 0.0
        self._current_tps_sum = 0.0
        self._current_tps_count = 0

    async def get_realtime_metrics(self) -> list[dict]:
        """Get per-second metrics for the last 60 seconds."""
        async with self._lock:
            # Flush current data first
            now = int(time.time())
            if now != self._current_second:
                await self._flush_current_second()
                self._current_second = now

            # Build result with gaps filled
            result = []
            buffer_dict = {int(p.timestamp): p for p in self._realtime_buffer}

            for i in range(60):
                ts = now - 59 + i
                if ts in buffer_dict:
                    p = buffer_dict[ts]
                    result.append({
                        "timestamp": ts,
                        "requests": p.requests,
                        "prompt_tokens": p.prompt_tokens,
                        "completion_tokens": p.completion_tokens,
                        "tps_min": round(p.tps_min, 1) if p.tps_count > 0 else 0,
                        "tps_max": round(p.tps_max, 1),
                        "tps_avg": round(p.tps_avg, 1),
                    })
                else:
                    result.append({
                        "timestamp": ts,
                        "requests": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "tps_min": 0,
                        "tps_max": 0,
                        "tps_avg": 0,
                    })

            return result

    async def get_historical_metrics(self, minutes: int = 60) -> list[dict]:
        """Get per-minute metrics for the specified time window."""
        now = int(time.time())
        start_ts = ((now // 60) - minutes + 1) * 60

        async with self._db.execute(
            """SELECT timestamp, requests, prompt_tokens, completion_tokens, tps_min, tps_max, tps_sum, tps_count
               FROM metrics_minute
               WHERE timestamp >= ?
               ORDER BY timestamp""",
            (start_ts,)
        ) as cursor:
            rows = await cursor.fetchall()

        # Build result with gaps filled
        result = []
        row_dict = {row[0]: row for row in rows}

        for i in range(minutes):
            ts = start_ts + (i * 60)
            if ts in row_dict:
                row = row_dict[ts]
                tps_count = row[7] or 0
                tps_avg = row[6] / tps_count if tps_count > 0 else 0
                result.append({
                    "timestamp": ts,
                    "requests": row[1],
                    "prompt_tokens": row[2],
                    "completion_tokens": row[3],
                    "tps_min": round(row[4] or 0, 1),
                    "tps_max": round(row[5] or 0, 1),
                    "tps_avg": round(tps_avg, 1),
                })
            else:
                result.append({
                    "timestamp": ts,
                    "requests": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "tps_min": 0,
                    "tps_max": 0,
                    "tps_avg": 0,
                })

        return result

    async def get_summary(self) -> dict:
        """Get summary statistics."""
        now = int(time.time())
        hour_ago = ((now // 60) - 60) * 60
        day_ago = ((now // 60) - 1440) * 60

        async with self._db.execute(
            """SELECT SUM(requests), SUM(prompt_tokens), SUM(completion_tokens),
                      MIN(CASE WHEN tps_min > 0 THEN tps_min ELSE NULL END),
                      MAX(tps_max),
                      SUM(tps_sum), SUM(tps_count)
               FROM metrics_minute WHERE timestamp >= ?""",
            (hour_ago,)
        ) as cursor:
            hour_row = await cursor.fetchone()

        async with self._db.execute(
            """SELECT SUM(requests), SUM(prompt_tokens), SUM(completion_tokens),
                      MIN(CASE WHEN tps_min > 0 THEN tps_min ELSE NULL END),
                      MAX(tps_max),
                      SUM(tps_sum), SUM(tps_count)
               FROM metrics_minute WHERE timestamp >= ?""",
            (day_ago,)
        ) as cursor:
            day_row = await cursor.fetchone()

        hour_tps_count = hour_row[6] or 0
        hour_tps_avg = hour_row[5] / hour_tps_count if hour_tps_count > 0 else 0
        day_tps_count = day_row[6] or 0
        day_tps_avg = day_row[5] / day_tps_count if day_tps_count > 0 else 0

        return {
            "last_hour": {
                "requests": hour_row[0] or 0,
                "prompt_tokens": hour_row[1] or 0,
                "completion_tokens": hour_row[2] or 0,
                "tps_min": round(hour_row[3] or 0, 1),
                "tps_max": round(hour_row[4] or 0, 1),
                "tps_avg": round(hour_tps_avg, 1),
            },
            "last_24h": {
                "requests": day_row[0] or 0,
                "prompt_tokens": day_row[1] or 0,
                "completion_tokens": day_row[2] or 0,
                "tps_min": round(day_row[3] or 0, 1),
                "tps_max": round(day_row[4] or 0, 1),
                "tps_avg": round(day_tps_avg, 1),
            },
        }

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            # Flush any pending data
            async with self._lock:
                await self._flush_current_second()
            await self._db.close()
            self._db = None
