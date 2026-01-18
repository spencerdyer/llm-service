"""Backend manager for orchestrating LLM backends."""

import os
from typing import AsyncIterator, Optional

from llm_service.backends.base import (
    BaseBackend,
    CompletionRequest,
    CompletionResponse,
    StreamChunk,
)
from llm_service.config import settings
from llm_service.services.config_manager import ConfigManager
from llm_service.services.model_manager import ModelInfo, ModelManager, ModelStatus, ModelType


def get_model_memory_mb(model_path: str) -> float:
    """Estimate model memory usage based on file sizes."""
    total_bytes = 0
    path = model_path if os.path.isdir(model_path) else os.path.dirname(model_path)

    if os.path.exists(path):
        for root, dirs, files in os.walk(path):
            for f in files:
                if f.endswith(('.safetensors', '.bin', '.gguf', '.npz')):
                    total_bytes += os.path.getsize(os.path.join(root, f))

    return total_bytes / (1024 * 1024)  # Convert to MB


class LoadedModel:
    """Container for a loaded model and its backend."""

    def __init__(self, model: ModelInfo, backend: BaseBackend):
        self.model = model
        self.backend = backend
        self.memory_mb = get_model_memory_mb(str(model.local_path))

    @property
    def is_loaded(self) -> bool:
        return self.backend.is_loaded

    @property
    def queue_length(self) -> int:
        """Get queue length from backend if available."""
        if hasattr(self.backend, 'queue_length'):
            return self.backend.queue_length
        return 0

    @property
    def is_queue_full(self) -> bool:
        """Check if backend's queue is full."""
        if hasattr(self.backend, 'is_queue_full'):
            return self.backend.is_queue_full
        return False


class BackendManager:
    """Manages LLM backend lifecycle and inference."""

    def __init__(self, model_manager: ModelManager, config_manager: ConfigManager):
        self.model_manager = model_manager
        self.config_manager = config_manager
        self._loaded_models: dict[str, LoadedModel] = {}
        self._active_model_id: Optional[str] = None

    @property
    def is_ready(self) -> bool:
        """Check if an active model is loaded and ready."""
        if not self._active_model_id:
            return False
        loaded = self._loaded_models.get(self._active_model_id)
        return loaded is not None and loaded.is_loaded

    @property
    def current_model(self) -> Optional[ModelInfo]:
        """Get the currently active model."""
        if self._active_model_id:
            loaded = self._loaded_models.get(self._active_model_id)
            if loaded:
                return loaded.model
        return None

    @property
    def loaded_models(self) -> list[ModelInfo]:
        """Get all loaded models."""
        return [lm.model for lm in self._loaded_models.values()]

    def _create_backend(self, model: ModelInfo) -> BaseBackend:
        """Create the appropriate backend for the model."""
        if settings.is_mac:
            # Use MLX on Mac (with batching support)
            from llm_service.backends.mlx_backend import MLXBackend

            return MLXBackend(str(model.local_path))
        else:
            # Use vLLM on Linux
            from llm_service.backends.vllm_backend import VLLMBackend

            return VLLMBackend(str(model.local_path), quantization=model.quantization)

    async def load_model(self, model_id_or_source: str) -> ModelInfo:
        """Load a model by ID or HuggingFace source.

        Args:
            model_id_or_source: Either a local model ID or HuggingFace repo ID

        Returns:
            The loaded model info
        """
        # First try to find by ID
        model = await self.model_manager.get_model(model_id_or_source)

        # Try by source
        if not model:
            model = await self.model_manager.get_model_by_source(model_id_or_source)

        if not model:
            raise ValueError(
                f"Model '{model_id_or_source}' not found. Download it first via the admin panel."
            )

        if model.status != ModelStatus.READY:
            raise ValueError(f"Model '{model.name}' is not ready (status: {model.status})")

        # Check if already loaded
        if model.id in self._loaded_models:
            self._active_model_id = model.id
            return model

        # Create and load backend
        backend = self._create_backend(model)
        await backend.load()

        loaded = LoadedModel(model, backend)
        self._loaded_models[model.id] = loaded
        self._active_model_id = model.id

        # Save as default
        await self.config_manager.set_setting("default_model", model.id)

        return model

    async def unload_model(self, model_id: Optional[str] = None) -> None:
        """Unload a model. If model_id is None, unload the active model."""
        target_id = model_id or self._active_model_id

        if not target_id:
            return

        loaded = self._loaded_models.get(target_id)
        if loaded:
            await loaded.backend.unload()
            del self._loaded_models[target_id]

            # If we unloaded the active model, select another or clear
            if self._active_model_id == target_id:
                if self._loaded_models:
                    self._active_model_id = next(iter(self._loaded_models.keys()))
                else:
                    self._active_model_id = None

    async def unload_all_models(self) -> None:
        """Unload all loaded models."""
        for model_id in list(self._loaded_models.keys()):
            await self.unload_model(model_id)

    def set_active_model(self, model_id: str) -> bool:
        """Set the active model for generation."""
        if model_id in self._loaded_models:
            self._active_model_id = model_id
            return True
        return False

    def check_queue_available(self, model_id: Optional[str] = None) -> tuple[bool, int]:
        """Check if the request queue can accept a new request.

        Returns:
            Tuple of (is_available, current_queue_length)
        """
        target_id = model_id or self._active_model_id
        if not target_id:
            return False, 0

        loaded = self._loaded_models.get(target_id)
        if not loaded:
            return False, 0

        return not loaded.is_queue_full, loaded.queue_length

    async def generate(self, request: CompletionRequest, model_id: Optional[str] = None) -> CompletionResponse:
        """Generate a completion using the specified or active model.

        The backend handles batching internally for concurrent requests.
        """
        target_id = model_id or self._active_model_id

        if not target_id:
            raise RuntimeError("No model loaded. Load a model first.")

        loaded = self._loaded_models.get(target_id)
        if not loaded or not loaded.is_loaded:
            raise RuntimeError(f"Model '{target_id}' is not loaded.")

        # Backend handles its own request queuing/batching
        return await loaded.backend.generate(request)

    async def generate_stream(
        self, request: CompletionRequest, model_id: Optional[str] = None
    ) -> AsyncIterator[StreamChunk]:
        """Generate a streaming completion using the specified or active model."""
        target_id = model_id or self._active_model_id

        if not target_id:
            raise RuntimeError("No model loaded. Load a model first.")

        loaded = self._loaded_models.get(target_id)
        if not loaded or not loaded.is_loaded:
            raise RuntimeError(f"Model '{target_id}' is not loaded.")

        # Backend handles streaming (note: streaming is still sequential for now)
        async for chunk in loaded.backend.generate_stream(request):
            yield chunk

    def get_loaded_models_info(self) -> list[dict]:
        """Get information about all loaded models."""
        result = []
        for model_id, loaded in self._loaded_models.items():
            info = {
                "id": loaded.model.id,
                "name": loaded.model.name,
                "source": loaded.model.source,
                "model_type": loaded.model.model_type.value,
                "quantization": loaded.model.quantization,
                "memory_mb": round(loaded.memory_mb, 1),
                "is_active": model_id == self._active_model_id,
                "queue_length": loaded.queue_length,
            }
            # Add backend-specific info
            backend_info = loaded.backend.get_model_info()
            if backend_info.get("batching_enabled"):
                info["batching_enabled"] = True
                info["max_batch_size"] = backend_info.get("max_batch_size", 1)
            result.append(info)
        return result

    def get_total_memory_mb(self) -> float:
        """Get total memory used by all loaded models."""
        return sum(lm.memory_mb for lm in self._loaded_models.values())

    def get_status(self) -> dict:
        """Get the current status of the backend."""
        loaded_models = self.get_loaded_models_info()

        status = {
            "ready": self.is_ready,
            "platform": settings.platform.value,
            "backend_type": "mlx" if settings.is_mac else "vllm",
            "current_model": None,
            "model_info": None,
            "loaded_models": loaded_models,
            "loaded_count": len(loaded_models),
            "total_memory_mb": round(self.get_total_memory_mb(), 1),
        }

        if self._active_model_id and self._active_model_id in self._loaded_models:
            loaded = self._loaded_models[self._active_model_id]
            status["current_model"] = {
                "id": loaded.model.id,
                "name": loaded.model.name,
                "source": loaded.model.source,
                "model_type": loaded.model.model_type.value,
                "quantization": loaded.model.quantization,
            }
            status["model_info"] = loaded.backend.get_model_info()

        return status

    async def shutdown(self) -> None:
        """Shutdown the backend manager."""
        await self.unload_all_models()
