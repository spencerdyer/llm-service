"""Workbench training and model-edit orchestration."""

from __future__ import annotations

import shutil
import subprocess
import sys
import uuid
import json
from pathlib import Path
from typing import Optional

from llm_service.config import settings
from llm_service.services.backend_manager import BackendManager, get_model_memory_mb
from llm_service.services.config_manager import ConfigManager
from llm_service.services.model_manager import ModelInfo, ModelManager, ModelStatus


class TrainingManager:
    """Coordinates workbench mode and long-running model edit jobs."""

    def __init__(self, config_manager: ConfigManager, model_manager: ModelManager):
        self.config_manager = config_manager
        self.model_manager = model_manager
        self.workbench_dir = settings.data_dir / "workbench"
        self.jobs_dir = self.workbench_dir / "jobs"

    async def initialize(self) -> None:
        """Ensure workbench storage exists."""
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

    async def get_operation_mode(self) -> str:
        """Return the persisted operation mode."""
        return await self.config_manager.get_operation_mode()

    def _get_model_compatibility_issue(self, model: ModelInfo) -> Optional[str]:
        """Return a user-facing reason when a model is not workbench-compatible."""
        if model.status != ModelStatus.READY:
            return f"Model status must be ready, got {model.status.value}"
        if model.metadata.get("experimental"):
            return "Experimental models cannot be used as new base models"
        if not model.local_path or not model.local_path.exists():
            return "Model files are missing from disk"
        if model.model_type.value == "gguf":
            return "GGUF models are not supported by the current abliteration runner"

        model_path = model.local_path
        if model_path.is_file():
            return "Single-file model artifacts are not supported"

        files = {path.name for path in model_path.iterdir() if path.is_file()}
        has_config = "config.json" in files
        has_weights = any(name.endswith((".safetensors", ".bin", ".npz")) for name in files)
        if not (has_config and has_weights):
            return "Model directory must include config and weight files"

        config_path = model_path / "config.json"
        if config_path.exists():
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return "Model config could not be parsed"

            architectures = config.get("architectures") or []
            model_type = str(config.get("model_type") or "").lower()
            is_multimodal = any(
                key in config
                for key in ("vision_config", "audio_config", "image_token_id", "video_token_id")
            ) or any("ConditionalGeneration" in arch for arch in architectures)
            if is_multimodal:
                return "Workbench abliteration currently supports text-only models only"

            if settings.is_mac and model_type in {"gemma4"}:
                return "Gemma 4 models are not supported by the current MLX workbench runner"

        return None

    async def switch_mode(self, mode: str, backend_manager: BackendManager) -> str:
        """Switch between inference and workbench modes."""
        if mode not in {"inference", "workbench"}:
            raise ValueError("Mode must be 'inference' or 'workbench'")

        current_mode = await self.get_operation_mode()
        if mode == current_mode:
            return mode

        if mode == "workbench":
            await backend_manager.unload_all_models()
        else:
            if await self.config_manager.has_running_workbench_jobs():
                raise ValueError("Wait for workbench jobs to finish before returning to inference mode")
            if await self.config_manager.has_running_evaluations():
                raise ValueError("Wait for evaluations to finish before returning to inference mode")

        await self.config_manager.set_operation_mode(mode)
        return mode

    def is_model_compatible(self, model: ModelInfo) -> bool:
        """Check whether a model can be copied and edited in the workbench."""
        return self._get_model_compatibility_issue(model) is None

    async def list_compatible_base_models(self) -> list[ModelInfo]:
        """List non-experimental ready models that the workbench can edit."""
        models = await self.model_manager.list_models(status=ModelStatus.READY)
        return [model for model in models if self.is_model_compatible(model)]

    def _job_dir(self, job_id: str) -> Path:
        return self.jobs_dir / job_id

    def _job_log_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "job.log"

    def _generate_derived_model_id(self, base_model_id: str) -> str:
        return f"{base_model_id}--abliterated-{uuid.uuid4().hex[:8]}"

    async def _ensure_disk_space(self, base_model: ModelInfo) -> None:
        """Fail early if there is not enough disk space for a safe copy."""
        required_mb = get_model_memory_mb(str(base_model.local_path))
        required_bytes = int(required_mb * 1024 * 1024 * 1.2)
        usage = shutil.disk_usage(self.model_manager.models_dir)
        if usage.free < required_bytes:
            raise ValueError(
                "Not enough free disk space to create a copied workbench model. "
                f"Need about {required_mb:.0f} MB plus headroom."
            )

    async def start_copy_abliterate_job(
        self,
        base_model_id: str,
        derived_name: Optional[str] = None,
        strength: float = 1.0,
        prompt_count: int = 6,
    ) -> dict:
        """Create a copied experimental model and launch the abliteration job."""
        if await self.get_operation_mode() != "workbench":
            raise ValueError("Switch the app to workbench mode before starting an abliteration job")

        if await self.config_manager.has_running_workbench_jobs():
            raise ValueError("Only one workbench job can run at a time")
        if await self.config_manager.has_running_evaluations():
            raise ValueError("Wait for active evaluations to finish before starting a job")

        base_model = await self.model_manager.get_model(base_model_id)
        if not base_model:
            raise ValueError("Base model not found")
        compatibility_issue = self._get_model_compatibility_issue(base_model)
        if compatibility_issue:
            raise ValueError(compatibility_issue)

        await self._ensure_disk_space(base_model)

        job_id = f"job-{uuid.uuid4().hex[:12]}"
        derived_model_id = self._generate_derived_model_id(base_model.id)
        display_name = derived_name.strip() if derived_name else f"{base_model.name} Abliterated"
        derived_path = self.model_manager.models_dir / derived_model_id
        log_path = self._job_log_path(job_id)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        await self.model_manager.create_derived_model(
            base_model=base_model,
            model_id=derived_model_id,
            name=display_name,
            derivation_type="abliteration",
            local_path=derived_path,
            job_id=job_id,
        )

        metadata = {
            "strength": strength,
            "prompt_count": prompt_count,
            "derived_name": display_name,
        }
        await self.config_manager.create_workbench_job(
            {
                "id": job_id,
                "job_type": "copy_abliterate",
                "status": "pending",
                "base_model_id": base_model.id,
                "derived_model_id": derived_model_id,
                "mode_snapshot": "workbench",
                "log_path": str(log_path),
                "progress": {"stage": "queued", "percent": 0},
                "metadata": metadata,
            }
        )

        try:
            with log_path.open("a", encoding="utf-8") as log_file:
                process = subprocess.Popen(
                    [sys.executable, "-m", "llm_service.workbench.job_runner", "--job-id", job_id],
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    cwd=str(Path.cwd()),
                )
        except Exception as exc:
            await self.config_manager.update_workbench_job(
                job_id,
                {"status": "error", "error": str(exc), "progress": {"stage": "failed", "percent": 0}},
            )
            await self.model_manager.mark_model_error(derived_model_id, str(exc))
            raise

        await self.config_manager.update_workbench_job(
            job_id,
            {
                "status": "running",
                "progress": {"stage": "starting", "percent": 2},
                "metadata": {**metadata, "pid": process.pid},
            },
        )
        return await self.get_job(job_id)

    async def get_job(self, job_id: str) -> Optional[dict]:
        """Fetch a workbench job with a short log tail."""
        job = await self.config_manager.get_workbench_job(job_id)
        if not job:
            return None
        job["log_tail"] = self._read_log_tail(job.get("log_path"))
        return job

    async def list_jobs(self) -> list[dict]:
        """List recent workbench jobs with short log excerpts."""
        jobs = await self.config_manager.list_workbench_jobs()
        for job in jobs:
            job["log_tail"] = self._read_log_tail(job.get("log_path"))
        return jobs

    async def promote_model(self, model_id: str) -> dict:
        """Promote an experimental workbench model for normal inference use."""
        if await self.get_operation_mode() != "workbench":
            raise ValueError("Promotion is only available in workbench mode")

        model = await self.model_manager.get_model(model_id)
        if not model:
            raise ValueError("Model not found")
        if model.status != ModelStatus.READY:
            raise ValueError("Only ready models can be promoted")
        if not model.metadata.get("experimental"):
            raise ValueError("This model is already available for inference")

        promoted = await self.model_manager.promote_model(model_id)
        return promoted.to_dict()

    def _read_log_tail(self, log_path: Optional[str], max_lines: int = 60) -> list[str]:
        """Read the tail of a workbench log file."""
        if not log_path:
            return []
        path = Path(log_path)
        if not path.exists():
            return []
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return []
        return lines[-max_lines:]

    async def get_dashboard_payload(self) -> dict:
        """Return workbench-specific dashboard data."""
        base_models = [model.to_dict() for model in await self.list_compatible_base_models()]
        jobs = await self.list_jobs()
        experimental_models = [
            model.to_dict()
            for model in await self.model_manager.list_models(status=ModelStatus.READY)
            if model.metadata.get("experimental")
        ]
        return {
            "operation_mode": await self.get_operation_mode(),
            "base_models": base_models,
            "jobs": jobs,
            "experimental_models": experimental_models,
        }
