"""Metrics collection and storage service."""

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import aiosqlite

from llm_service.config import settings


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

        await self._db.commit()

    async def record_request(self, prompt_tokens: int = 0, completion_tokens: int = 0, tokens_per_second: float = 0.0) -> None:
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
