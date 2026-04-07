"""Standalone subprocess for quantization workbench jobs using llm-compressor."""

from __future__ import annotations

import argparse

import json
import sqlite3
from pathlib import Path

from llm_service.config import settings

SUPPORTED_METHODS = {"gptq", "nvfp4"}

DEFAULT_CALIBRATION_SAMPLES = {
    "gptq": 512,
    "nvfp4": 128,
}

DEFAULT_MAX_SEQ_LENGTH = 2048


def connect_db() -> sqlite3.Connection:
    conn = sqlite3.connect(settings.effective_db_path)
    conn.row_factory = sqlite3.Row
    return conn


def fetch_workbench_job(conn: sqlite3.Connection, job_id: str) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM workbench_jobs WHERE id = ?", (job_id,)).fetchone()
    if row is None:
        raise RuntimeError(f"Workbench job '{job_id}' not found")
    return row


def fetch_model(conn: sqlite3.Connection, model_id: str) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
    if row is None:
        raise RuntimeError(f"Model '{model_id}' not found")
    return row


def decode_json(value: str | None, default):
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def update_job(
    conn: sqlite3.Connection,
    job_id: str,
    *,
    status: str | None = None,
    progress: dict | None = None,
    error: str | None = None,
    metadata: dict | None = None,
) -> None:
    current = fetch_workbench_job(conn, job_id)
    merged_metadata = decode_json(current["metadata"], {})
    if metadata:
        merged_metadata.update(metadata)

    conn.execute(
        """
        UPDATE workbench_jobs
        SET status = COALESCE(?, status),
            progress = COALESCE(?, progress),
            error = ?,
            metadata = ?,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (
            status,
            json.dumps(progress) if progress is not None else None,
            error,
            json.dumps(merged_metadata),
            job_id,
        ),
    )
    conn.commit()


def update_model_status(
    conn: sqlite3.Connection,
    model_id: str,
    *,
    status: str | None = None,
    metadata_updates: dict | None = None,
) -> None:
    current = fetch_model(conn, model_id)
    model_metadata = decode_json(current["metadata"], {})
    if metadata_updates:
        model_metadata.update(metadata_updates)

    conn.execute(
        """
        UPDATE models
        SET status = COALESCE(?, status),
            quantization = COALESCE(quantization, ?),
            model_type = COALESCE(model_type, ?),
            metadata = ?,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (
            status,
            metadata_updates.get("quantization") if metadata_updates else None,
            metadata_updates.get("model_type") if metadata_updates else None,
            json.dumps(model_metadata),
            model_id,
        ),
    )
    conn.commit()


def _load_calibration_dataset(
    tokenizer, num_samples: int, max_seq_length: int
) -> "Dataset":
    """Load and tokenize a calibration dataset for quantization."""
    from datasets import load_dataset

    ds = load_dataset(
        "HuggingFaceH4/ultrachat_200k",
        split=f"train_sft[:{num_samples}]",
    )
    ds = ds.shuffle(seed=42)

    def preprocess(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"], tokenize=False
            )
        }

    ds = ds.map(preprocess)

    def tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=max_seq_length,
            truncation=True,
            add_special_tokens=False,
        )

    ds = ds.map(tokenize, remove_columns=ds.column_names)
    return ds


def _get_ignore_layers(base_model_path: Path) -> list[str]:
    """Build the list of layers to exclude from quantization.

    Always ignores lm_head. For multimodal models (detected via config),
    also ignores vision/projection layers that vLLM loads as unquantized.
    """
    ignore = ["lm_head"]

    config_path = base_model_path / "config.json"
    if not config_path.exists():
        return ignore

    config = json.loads(config_path.read_text(encoding="utf-8"))

    is_multimodal = any(
        key in config
        for key in ("vision_config", "visual", "mm_projector_type",
                     "image_tower", "video_tower")
    )
    if is_multimodal:
        ignore.extend([
            "re:.*vision_tower.*",
            "re:.*visual.*",
            "re:.*embed_vision.*",
            "re:.*multi_modal_projector.*",
            "re:.*mm_projector.*",
            "re:.*image_newline.*",
        ])
        print(f"Multimodal model detected — excluding from quantization: {ignore}")

    return ignore


def run_gptq_quantization(
    base_model_path: Path,
    output_path: Path,
    num_calibration_samples: int,
    max_seq_length: int,
) -> dict:
    """Run GPTQ W4A16 quantization via llm-compressor."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from llmcompressor import oneshot
    from llmcompressor.modifiers.gptq import GPTQModifier

    print(f"Loading model from {base_model_path} for GPTQ quantization")
    model = AutoModelForCausalLM.from_pretrained(
        str(base_model_path), dtype="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(base_model_path), trust_remote_code=True
    )

    print(f"Preparing calibration dataset ({num_calibration_samples} samples)...")
    ds = _load_calibration_dataset(tokenizer, num_calibration_samples, max_seq_length)

    ignore = _get_ignore_layers(base_model_path)
    recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=ignore)

    print("Running GPTQ oneshot quantization...")
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=max_seq_length,
        num_calibration_samples=num_calibration_samples,
    )

    print(f"Saving quantized model to {output_path}...")
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_path), save_compressed=True)
    tokenizer.save_pretrained(str(output_path))
    del model, tokenizer, ds

    _copy_extra_files(base_model_path, output_path)

    return _compute_summary(
        base_model_path,
        output_path,
        method="gptq",
        scheme="W4A16",
        num_calibration_samples=num_calibration_samples,
    )


def run_nvfp4_quantization(
    base_model_path: Path,
    output_path: Path,
    num_calibration_samples: int,
    max_seq_length: int,
) -> dict:
    """Run NVFP4 quantization via llm-compressor."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    print(f"Loading model from {base_model_path} for NVFP4 quantization")
    model = AutoModelForCausalLM.from_pretrained(
        str(base_model_path), dtype="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(base_model_path), trust_remote_code=True
    )

    print(f"Preparing calibration dataset ({num_calibration_samples} samples)...")
    ds = _load_calibration_dataset(tokenizer, num_calibration_samples, max_seq_length)

    ignore = _get_ignore_layers(base_model_path)
    recipe = QuantizationModifier(targets="Linear", scheme="NVFP4", ignore=ignore)

    print("Running NVFP4 oneshot quantization...")
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=max_seq_length,
        num_calibration_samples=num_calibration_samples,
    )

    print(f"Saving quantized model to {output_path}...")
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_path), save_compressed=True)
    tokenizer.save_pretrained(str(output_path))
    del model, tokenizer, ds

    _copy_extra_files(base_model_path, output_path)

    return _compute_summary(
        base_model_path,
        output_path,
        method="nvfp4",
        scheme="NVFP4",
        num_calibration_samples=num_calibration_samples,
    )


def _copy_extra_files(base_model_path: Path, output_path: Path) -> None:
    """Copy auxiliary files that aren't saved by save_pretrained."""
    import shutil

    for name in ("processor_config.json", "chat_template.jinja"):
        src = base_model_path / name
        if src.exists():
            shutil.copy2(src, output_path / name)


def _compute_summary(
    base_model_path: Path,
    output_path: Path,
    *,
    method: str,
    scheme: str,
    num_calibration_samples: int,
) -> dict:
    weight_exts = (".safetensors", ".bin")
    base_size = sum(
        f.stat().st_size
        for f in base_model_path.rglob("*")
        if f.is_file() and f.suffix in weight_exts
    )
    quant_size = sum(
        f.stat().st_size
        for f in output_path.rglob("*")
        if f.is_file() and f.suffix in weight_exts
    )
    return {
        "method": method,
        "scheme": scheme,
        "num_calibration_samples": num_calibration_samples,
        "base_size_mb": round(base_size / (1024 * 1024), 1),
        "quantized_size_mb": round(quant_size / (1024 * 1024), 1),
        "compression_ratio": round(base_size / max(quant_size, 1), 2),
    }


def _free_gpu_memory() -> None:
    """Release all GPU memory so the smoke test can load the quantized model."""
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def smoke_test_model(model_path: str) -> None:
    """Validate the quantized model files without loading onto GPU."""
    from pathlib import Path

    model_dir = Path(model_path)
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise RuntimeError("Quantized model is missing config.json")

    import json

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if "model_type" not in config:
        raise RuntimeError("Quantized model config.json is missing model_type")

    weight_files = [
        f for f in model_dir.iterdir()
        if f.suffix in (".safetensors", ".bin") and f.stat().st_size > 0
    ]
    if not weight_files:
        raise RuntimeError("Quantized model has no weight files")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.vocab_size == 0:
        raise RuntimeError("Tokenizer loaded but has empty vocabulary")

    print(
        f"Smoke test passed: config OK, {len(weight_files)} weight file(s), "
        f"tokenizer vocab size {tokenizer.vocab_size}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()

    conn = connect_db()
    try:
        job = fetch_workbench_job(conn, args.job_id)
        base_model = fetch_model(conn, job["base_model_id"])
        derived_model = fetch_model(conn, job["derived_model_id"])

        base_path = Path(base_model["local_path"])
        output_path = Path(derived_model["local_path"])
        job_metadata = decode_json(job["metadata"], {})

        method = job_metadata.get("method", "gptq")
        if method not in SUPPORTED_METHODS:
            raise RuntimeError(f"Unsupported quantization method: {method}")

        num_calibration_samples = int(
            job_metadata.get(
                "num_calibration_samples", DEFAULT_CALIBRATION_SAMPLES[method]
            )
        )
        max_seq_length = int(
            job_metadata.get("max_seq_length", DEFAULT_MAX_SEQ_LENGTH)
        )

        print(f"Starting {method.upper()} quantization job {args.job_id}")
        update_job(
            conn,
            args.job_id,
            status="running",
            progress={"stage": "loading_model", "percent": 5},
        )

        if output_path.exists():
            raise RuntimeError(f"Output path already exists: {output_path}")

        update_job(
            conn,
            args.job_id,
            progress={"stage": "quantizing", "percent": 15},
        )

        if method == "gptq":
            summary = run_gptq_quantization(
                base_model_path=base_path,
                output_path=output_path,
                num_calibration_samples=num_calibration_samples,
                max_seq_length=max_seq_length,
            )
        elif method == "nvfp4":
            summary = run_nvfp4_quantization(
                base_model_path=base_path,
                output_path=output_path,
                num_calibration_samples=num_calibration_samples,
                max_seq_length=max_seq_length,
            )

        print(f"Quantization summary: {json.dumps(summary)}")

        update_job(
            conn,
            args.job_id,
            progress={"stage": "validating", "percent": 85},
        )
        _free_gpu_memory()
        smoke_test_model(str(output_path))

        update_model_status(
            conn,
            derived_model["id"],
            status="ready",
            metadata_updates={
                "experimental": True,
                "promotion_state": "experimental",
                "quantization": method,
                "model_type": "vllm",
                "quantization_summary": summary,
            },
        )
        update_job(
            conn,
            args.job_id,
            status="complete",
            progress={"stage": "complete", "percent": 100},
            metadata={"summary": summary},
        )
        print("Quantization job complete")
        return 0
    except Exception as exc:
        print(f"Quantization job failed: {exc}")
        try:
            derived_model_id = fetch_workbench_job(conn, args.job_id)[
                "derived_model_id"
            ]
            update_model_status(
                conn,
                derived_model_id,
                status="error",
                metadata_updates={"error": str(exc)},
            )
        except Exception:
            pass
        update_job(
            conn,
            args.job_id,
            status="error",
            progress={"stage": "failed", "percent": 100},
            error=str(exc),
        )
        return 1
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
