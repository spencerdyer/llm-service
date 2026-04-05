"""Standalone subprocess for workbench model evaluations."""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3

from llm_service.backends.base import CompletionRequest, GenerationConfig
from llm_service.config import settings


def connect_db() -> sqlite3.Connection:
    conn = sqlite3.connect(settings.effective_db_path)
    conn.row_factory = sqlite3.Row
    return conn


def decode_json(value: str | None, default):
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def fetch_evaluation(conn: sqlite3.Connection, evaluation_id: str) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM evaluation_runs WHERE id = ?", (evaluation_id,)).fetchone()
    if row is None:
        raise RuntimeError(f"Evaluation run '{evaluation_id}' not found")
    return row


def fetch_model(conn: sqlite3.Connection, model_id: str) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
    if row is None:
        raise RuntimeError(f"Model '{model_id}' not found")
    return row


def update_evaluation(
    conn: sqlite3.Connection,
    evaluation_id: str,
    *,
    status: str | None = None,
    results: dict | None = None,
    error: str | None = None,
    metadata: dict | None = None,
) -> None:
    current = fetch_evaluation(conn, evaluation_id)
    merged_metadata = decode_json(current["metadata"], {})
    if metadata:
        merged_metadata.update(metadata)

    conn.execute(
        """
        UPDATE evaluation_runs
        SET status = COALESCE(?, status),
            results = COALESCE(?, results),
            error = ?,
            metadata = ?,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (
            status,
            json.dumps(results) if results is not None else None,
            error,
            json.dumps(merged_metadata),
            evaluation_id,
        ),
    )
    conn.commit()


async def build_backend(model_row: sqlite3.Row):
    if settings.is_mac:
        from llm_service.backends.mlx_backend import MLXBackend

        backend = MLXBackend(model_row["local_path"])
    else:
        from llm_service.backends.vllm_backend import VLLMBackend

        backend = VLLMBackend(model_row["local_path"], quantization=model_row["quantization"])
    await backend.load()
    return backend


async def evaluate_model(model_row: sqlite3.Row, prompts: list[str]) -> dict:
    backend = await build_backend(model_row)
    outputs = []
    try:
        for prompt in prompts:
            response = await backend.generate(
                CompletionRequest(
                    prompt=prompt,
                    config=GenerationConfig(max_tokens=256, temperature=0.2, top_p=0.9),
                    stream=False,
                )
            )
            outputs.append(
                {
                    "prompt": prompt,
                    "response": response.text,
                    "prompt_tokens": response.prompt_tokens,
                    "completion_tokens": response.completion_tokens,
                }
            )
    finally:
        await backend.unload()

    total_completion_tokens = sum(item["completion_tokens"] for item in outputs)
    return {
        "model_id": model_row["id"],
        "model_name": model_row["name"],
        "outputs": outputs,
        "summary": {
            "prompt_count": len(outputs),
            "completion_tokens": total_completion_tokens,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-id", required=True)
    args = parser.parse_args()

    conn = connect_db()
    try:
        evaluation = fetch_evaluation(conn, args.evaluation_id)
        prompts = decode_json(evaluation["prompt_input"], [])
        if not prompts:
            raise RuntimeError("No prompts were supplied for this evaluation")

        print(f"Starting evaluation {args.evaluation_id}")
        primary_model = fetch_model(conn, evaluation["model_id"])
        baseline_model = (
            fetch_model(conn, evaluation["baseline_model_id"])
            if evaluation["baseline_model_id"]
            else None
        )

        primary_results = asyncio.run(evaluate_model(primary_model, prompts))
        results = {
            "primary": primary_results,
            "baseline": None,
        }

        if baseline_model:
            print(f"Evaluating baseline model {baseline_model['id']}")
            results["baseline"] = asyncio.run(evaluate_model(baseline_model, prompts))

        update_evaluation(conn, args.evaluation_id, status="complete", results=results)
        print("Evaluation complete")
        return 0
    except Exception as exc:
        print(f"Evaluation failed: {exc}")
        update_evaluation(conn, args.evaluation_id, status="error", error=str(exc))
        return 1
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
