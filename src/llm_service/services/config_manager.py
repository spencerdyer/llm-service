"""Configuration and settings persistence."""

import json
from pathlib import Path
from typing import Any, Optional

import aiosqlite

from llm_service.config import settings


class ConfigManager:
    """Manages persistent configuration in SQLite."""

    def __init__(self):
        self.db_path = settings.effective_db_path
        self._db: Optional[aiosqlite.Connection] = None

    async def initialize(self) -> None:
        """Initialize the database."""
        self._db = await aiosqlite.connect(self.db_path)
        await self._create_tables()

    async def _create_tables(self) -> None:
        """Create necessary tables."""
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS models (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                source TEXT NOT NULL,
                local_path TEXT,
                model_type TEXT,
                quantization TEXT,
                status TEXT DEFAULT 'pending',
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Per-model configuration settings
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS model_config (
                model_id TEXT PRIMARY KEY,
                display_name TEXT,
                system_prompt TEXT,
                temperature REAL DEFAULT 0.7,
                max_tokens INTEGER DEFAULT 2048,
                top_p REAL DEFAULT 0.9,
                top_k INTEGER DEFAULT 50,
                repetition_penalty REAL DEFAULT 1.1,
                context_length INTEGER DEFAULT 4096,
                stop_sequences TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (model_id) REFERENCES models(id) ON DELETE CASCADE
            )
        """)

        await self._db.commit()

    async def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a setting value."""
        async with self._db.execute(
            "SELECT value FROM settings WHERE key = ?", (key,)
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                try:
                    return json.loads(row[0])
                except json.JSONDecodeError:
                    return row[0]
            return default

    async def set_setting(self, key: str, value: Any) -> None:
        """Set a setting value."""
        json_value = json.dumps(value)
        await self._db.execute(
            """
            INSERT INTO settings (key, value, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = CURRENT_TIMESTAMP
            """,
            (key, json_value),
        )
        await self._db.commit()

    async def delete_setting(self, key: str) -> None:
        """Delete a setting."""
        await self._db.execute("DELETE FROM settings WHERE key = ?", (key,))
        await self._db.commit()

    async def get_all_settings(self) -> dict[str, Any]:
        """Get all settings."""
        settings_dict = {}
        async with self._db.execute("SELECT key, value FROM settings") as cursor:
            async for row in cursor:
                try:
                    settings_dict[row[0]] = json.loads(row[1])
                except json.JSONDecodeError:
                    settings_dict[row[0]] = row[1]
        return settings_dict

    async def get_model_config(self, model_id: str) -> Optional[dict]:
        """Get configuration for a specific model."""
        async with self._db.execute(
            """SELECT display_name, system_prompt, temperature, max_tokens,
                      top_p, top_k, repetition_penalty, context_length, stop_sequences
               FROM model_config WHERE model_id = ?""",
            (model_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                stop_sequences = []
                if row[8]:
                    try:
                        stop_sequences = json.loads(row[8])
                    except json.JSONDecodeError:
                        stop_sequences = [s.strip() for s in row[8].split(",") if s.strip()]
                return {
                    "model_id": model_id,
                    "display_name": row[0],
                    "system_prompt": row[1],
                    "temperature": row[2],
                    "max_tokens": row[3],
                    "top_p": row[4],
                    "top_k": row[5],
                    "repetition_penalty": row[6],
                    "context_length": row[7],
                    "stop_sequences": stop_sequences,
                }
        return None

    async def set_model_config(self, model_id: str, config: dict) -> None:
        """Set configuration for a specific model."""
        stop_sequences = config.get("stop_sequences", [])
        if isinstance(stop_sequences, list):
            stop_sequences = json.dumps(stop_sequences)

        await self._db.execute(
            """
            INSERT INTO model_config (
                model_id, display_name, system_prompt, temperature, max_tokens,
                top_p, top_k, repetition_penalty, context_length, stop_sequences, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(model_id) DO UPDATE SET
                display_name = excluded.display_name,
                system_prompt = excluded.system_prompt,
                temperature = excluded.temperature,
                max_tokens = excluded.max_tokens,
                top_p = excluded.top_p,
                top_k = excluded.top_k,
                repetition_penalty = excluded.repetition_penalty,
                context_length = excluded.context_length,
                stop_sequences = excluded.stop_sequences,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                model_id,
                config.get("display_name"),
                config.get("system_prompt"),
                config.get("temperature", 0.7),
                config.get("max_tokens", 2048),
                config.get("top_p", 0.9),
                config.get("top_k", 50),
                config.get("repetition_penalty", 1.1),
                config.get("context_length", 4096),
                stop_sequences,
            ),
        )
        await self._db.commit()

    async def delete_model_config(self, model_id: str) -> None:
        """Delete configuration for a specific model."""
        await self._db.execute("DELETE FROM model_config WHERE model_id = ?", (model_id,))
        await self._db.commit()

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None
