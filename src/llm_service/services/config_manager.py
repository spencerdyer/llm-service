"""Configuration and settings persistence."""

import json
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
        self._db = await aiosqlite.connect(self.db_path, isolation_level=None)
        await self._db.execute("PRAGMA foreign_keys = ON")
        await self._db.execute("PRAGMA journal_mode = WAL")
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

        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS workbench_jobs (
                id TEXT PRIMARY KEY,
                job_type TEXT NOT NULL,
                status TEXT NOT NULL,
                base_model_id TEXT NOT NULL,
                derived_model_id TEXT,
                mode_snapshot TEXT,
                log_path TEXT,
                progress TEXT,
                error TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS evaluation_runs (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                model_id TEXT NOT NULL,
                baseline_model_id TEXT,
                prompt_suite TEXT,
                prompt_input TEXT,
                results TEXT,
                error TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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

    async def get_operation_mode(self) -> str:
        """Get the current operation mode."""
        return await self.get_setting("operation_mode", "inference")

    async def set_operation_mode(self, mode: str) -> None:
        """Persist the current operation mode."""
        await self.set_setting("operation_mode", mode)

    def _deserialize_json(self, value: Optional[str], default: Any) -> Any:
        """Decode JSON columns with a default fallback."""
        if not value:
            return default
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default

    async def create_workbench_job(self, job: dict[str, Any]) -> None:
        """Create a new workbench job."""
        await self._db.execute(
            """
            INSERT INTO workbench_jobs (
                id, job_type, status, base_model_id, derived_model_id, mode_snapshot,
                log_path, progress, error, metadata, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """,
            (
                job["id"],
                job["job_type"],
                job["status"],
                job["base_model_id"],
                job.get("derived_model_id"),
                job.get("mode_snapshot"),
                job.get("log_path"),
                json.dumps(job.get("progress", {})),
                job.get("error"),
                json.dumps(job.get("metadata", {})),
            ),
        )
        await self._db.commit()

    async def update_workbench_job(self, job_id: str, updates: dict[str, Any]) -> None:
        """Update fields on an existing workbench job."""
        if not updates:
            return

        assignments = []
        values = []
        json_fields = {"progress", "metadata"}
        for key, value in updates.items():
            assignments.append(f"{key} = ?")
            if key in json_fields:
                values.append(json.dumps(value))
            else:
                values.append(value)

        assignments.append("updated_at = CURRENT_TIMESTAMP")
        values.append(job_id)
        await self._db.execute(
            f"UPDATE workbench_jobs SET {', '.join(assignments)} WHERE id = ?",
            values,
        )
        await self._db.commit()

    async def get_workbench_job(self, job_id: str) -> Optional[dict[str, Any]]:
        """Fetch a single workbench job."""
        async with self._db.execute(
            """
            SELECT id, job_type, status, base_model_id, derived_model_id, mode_snapshot,
                   log_path, progress, error, metadata, created_at, updated_at
            FROM workbench_jobs
            WHERE id = ?
            """,
            (job_id,),
        ) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            return {
                "id": row[0],
                "job_type": row[1],
                "status": row[2],
                "base_model_id": row[3],
                "derived_model_id": row[4],
                "mode_snapshot": row[5],
                "log_path": row[6],
                "progress": self._deserialize_json(row[7], {}),
                "error": row[8],
                "metadata": self._deserialize_json(row[9], {}),
                "created_at": row[10],
                "updated_at": row[11],
            }
        return None

    async def list_workbench_jobs(self, limit: int = 50) -> list[dict[str, Any]]:
        """List recent workbench jobs."""
        jobs = []
        async with self._db.execute(
            """
            SELECT id, job_type, status, base_model_id, derived_model_id, mode_snapshot,
                   log_path, progress, error, metadata, created_at, updated_at
            FROM workbench_jobs
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ) as cursor:
            async for row in cursor:
                jobs.append(
                    {
                        "id": row[0],
                        "job_type": row[1],
                        "status": row[2],
                        "base_model_id": row[3],
                        "derived_model_id": row[4],
                        "mode_snapshot": row[5],
                        "log_path": row[6],
                        "progress": self._deserialize_json(row[7], {}),
                        "error": row[8],
                        "metadata": self._deserialize_json(row[9], {}),
                        "created_at": row[10],
                        "updated_at": row[11],
                    }
                )
        return jobs

    async def has_running_workbench_jobs(self) -> bool:
        """Check for any running workbench jobs."""
        async with self._db.execute(
            "SELECT 1 FROM workbench_jobs WHERE status IN ('pending', 'running') LIMIT 1"
        ) as cursor:
            return await cursor.fetchone() is not None

    async def fail_stale_workbench_jobs(self) -> None:
        """Mark interrupted workbench jobs as failed on startup."""
        await self._db.execute(
            """
            UPDATE workbench_jobs
            SET status = 'error',
                error = COALESCE(error, 'Workbench job interrupted by service restart'),
                progress = json_set(COALESCE(progress, '{}'), '$.stale_job', true),
                updated_at = CURRENT_TIMESTAMP
            WHERE status IN ('pending', 'running')
            """
        )
        await self._db.commit()

    async def create_evaluation_run(self, run: dict[str, Any]) -> None:
        """Create a new evaluation run."""
        await self._db.execute(
            """
            INSERT INTO evaluation_runs (
                id, status, model_id, baseline_model_id, prompt_suite, prompt_input,
                results, error, metadata, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """,
            (
                run["id"],
                run["status"],
                run["model_id"],
                run.get("baseline_model_id"),
                run.get("prompt_suite"),
                json.dumps(run.get("prompt_input", [])),
                json.dumps(run.get("results", {})),
                run.get("error"),
                json.dumps(run.get("metadata", {})),
            ),
        )
        await self._db.commit()

    async def update_evaluation_run(self, run_id: str, updates: dict[str, Any]) -> None:
        """Update fields on an evaluation run."""
        if not updates:
            return

        assignments = []
        values = []
        json_fields = {"prompt_input", "results", "metadata"}
        for key, value in updates.items():
            assignments.append(f"{key} = ?")
            if key in json_fields:
                values.append(json.dumps(value))
            else:
                values.append(value)

        assignments.append("updated_at = CURRENT_TIMESTAMP")
        values.append(run_id)
        await self._db.execute(
            f"UPDATE evaluation_runs SET {', '.join(assignments)} WHERE id = ?",
            values,
        )
        await self._db.commit()

    async def get_evaluation_run(self, run_id: str) -> Optional[dict[str, Any]]:
        """Fetch a single evaluation run."""
        async with self._db.execute(
            """
            SELECT id, status, model_id, baseline_model_id, prompt_suite, prompt_input,
                   results, error, metadata, created_at, updated_at
            FROM evaluation_runs
            WHERE id = ?
            """,
            (run_id,),
        ) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            return {
                "id": row[0],
                "status": row[1],
                "model_id": row[2],
                "baseline_model_id": row[3],
                "prompt_suite": row[4],
                "prompt_input": self._deserialize_json(row[5], []),
                "results": self._deserialize_json(row[6], {}),
                "error": row[7],
                "metadata": self._deserialize_json(row[8], {}),
                "created_at": row[9],
                "updated_at": row[10],
            }
        return None

    async def list_evaluation_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        """List recent evaluation runs."""
        runs = []
        async with self._db.execute(
            """
            SELECT id, status, model_id, baseline_model_id, prompt_suite, prompt_input,
                   results, error, metadata, created_at, updated_at
            FROM evaluation_runs
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ) as cursor:
            async for row in cursor:
                runs.append(
                    {
                        "id": row[0],
                        "status": row[1],
                        "model_id": row[2],
                        "baseline_model_id": row[3],
                        "prompt_suite": row[4],
                        "prompt_input": self._deserialize_json(row[5], []),
                        "results": self._deserialize_json(row[6], {}),
                        "error": row[7],
                        "metadata": self._deserialize_json(row[8], {}),
                        "created_at": row[9],
                        "updated_at": row[10],
                    }
                )
        return runs

    async def has_running_evaluations(self) -> bool:
        """Check for any in-flight evaluation runs."""
        async with self._db.execute(
            "SELECT 1 FROM evaluation_runs WHERE status IN ('pending', 'running') LIMIT 1"
        ) as cursor:
            return await cursor.fetchone() is not None

    async def fail_stale_evaluations(self) -> None:
        """Mark interrupted evaluation runs as failed on startup."""
        await self._db.execute(
            """
            UPDATE evaluation_runs
            SET status = 'error',
                error = COALESCE(error, 'Evaluation interrupted by service restart'),
                updated_at = CURRENT_TIMESTAMP
            WHERE status IN ('pending', 'running')
            """
        )
        await self._db.commit()

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
