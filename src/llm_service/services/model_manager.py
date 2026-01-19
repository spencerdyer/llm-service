"""Model management service."""

import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import AsyncIterator, Optional

import aiosqlite
from huggingface_hub import HfApi, snapshot_download

from llm_service.config import settings


class ModelStatus(str, Enum):
    PENDING = "pending"
    DOWNLOADING = "downloading"
    READY = "ready"
    ERROR = "error"


class ModelType(str, Enum):
    MLX = "mlx"
    VLLM = "vllm"
    GGUF = "gguf"
    UNKNOWN = "unknown"


@dataclass
class ModelInfo:
    """Information about a downloaded model."""

    id: str
    name: str
    source: str  # HuggingFace repo ID
    local_path: Optional[Path] = None
    model_type: ModelType = ModelType.UNKNOWN
    quantization: Optional[str] = None
    status: ModelStatus = ModelStatus.PENDING
    metadata: dict = field(default_factory=dict)

    @property
    def capabilities(self) -> dict:
        """Detect capabilities from model name and metadata."""
        # Check if capabilities are stored in metadata
        if "capabilities" in self.metadata:
            return self.metadata["capabilities"]

        # Auto-detect from name
        name_lower = self.name.lower()
        source_lower = self.source.lower()
        combined = f"{name_lower} {source_lower}"

        caps = {"thinking": False, "tool_use": False}

        # Thinking/reasoning patterns
        thinking_patterns = [
            "thinking", "reason", "-r1", "deepseek-r1", "qwq",
            "o1-", "o3-", "reflection", "cot", "chain-of-thought"
        ]
        if any(p in combined for p in thinking_patterns):
            caps["thinking"] = True

        # Tool use patterns
        tool_patterns = ["instruct", "chat", "tool", "function", "agent"]
        if any(p in combined for p in tool_patterns):
            caps["tool_use"] = True

        return caps

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "source": self.source,
            "local_path": str(self.local_path) if self.local_path else None,
            "model_type": self.model_type.value,
            "quantization": self.quantization,
            "status": self.status.value,
            "metadata": self.metadata,
            "capabilities": self.capabilities,
            "favorite": self.metadata.get("favorite", False),
        }

    @classmethod
    def from_row(cls, row: tuple) -> "ModelInfo":
        """Create from database row."""
        return cls(
            id=row[0],
            name=row[1],
            source=row[2],
            local_path=Path(row[3]) if row[3] else None,
            model_type=ModelType(row[4]) if row[4] else ModelType.UNKNOWN,
            quantization=row[5],
            status=ModelStatus(row[6]) if row[6] else ModelStatus.PENDING,
            metadata=json.loads(row[7]) if row[7] else {},
        )


class ModelManager:
    """Manages model downloads and storage."""

    def __init__(self, config_manager=None):
        self._config_manager = config_manager
        self._models_dir = settings.effective_models_dir
        self.db_path = settings.effective_db_path
        self._db: Optional[aiosqlite.Connection] = None
        self._hf_api = HfApi()
        self._download_progress: dict[str, float] = {}

    @property
    def models_dir(self) -> Path:
        """Get the current models directory."""
        return self._models_dir

    async def initialize(self) -> None:
        """Initialize the model manager."""
        self._db = await aiosqlite.connect(self.db_path)

        # Check for custom models directory in settings
        if self._config_manager:
            custom_dir = await self._config_manager.get_setting("models_dir")
            if custom_dir:
                self._models_dir = Path(custom_dir)

        self._models_dir.mkdir(parents=True, exist_ok=True)

        # Clean up stale downloads from previous sessions
        await self._cleanup_stale_downloads()

    async def _cleanup_stale_downloads(self) -> None:
        """Reset models stuck in 'downloading' status from previous sessions."""
        await self._db.execute(
            """
            UPDATE models
            SET status = ?, metadata = json_set(COALESCE(metadata, '{}'), '$.stale_download', true)
            WHERE status = ?
            """,
            (ModelStatus.ERROR.value, ModelStatus.DOWNLOADING.value),
        )
        await self._db.commit()

    async def set_models_dir(self, path: str) -> None:
        """Set the models directory and persist to settings."""
        new_path = Path(path).expanduser().resolve()
        new_path.mkdir(parents=True, exist_ok=True)
        self._models_dir = new_path

        if self._config_manager:
            await self._config_manager.set_setting("models_dir", str(new_path))

    async def close(self) -> None:
        """Close database connection."""
        if self._db:
            await self._db.close()
            self._db = None

    def _detect_model_type(self, repo_id: str, files: list[str]) -> tuple[ModelType, Optional[str]]:
        """Detect the model type and quantization from repository files."""
        repo_lower = repo_id.lower()
        files_lower = [f.lower() for f in files]

        # Check for MLX models
        if "mlx" in repo_lower or any("mlx" in f for f in files_lower):
            # Detect quantization
            quant = None
            if "4bit" in repo_lower or "4-bit" in repo_lower:
                quant = "4bit"
            elif "8bit" in repo_lower or "8-bit" in repo_lower:
                quant = "8bit"
            return ModelType.MLX, quant

        # Check for GGUF models
        if any(f.endswith(".gguf") for f in files):
            quant = None
            for q in ["q4_k_m", "q4_0", "q5_k_m", "q5_0", "q8_0", "q6_k"]:
                if q in repo_lower:
                    quant = q
                    break
            return ModelType.GGUF, quant

        # Check for quantized models (AWQ, GPTQ)
        if "awq" in repo_lower:
            return ModelType.VLLM, "awq"
        if "gptq" in repo_lower:
            return ModelType.VLLM, "gptq"

        # Default to vLLM for standard transformers models
        if any(f.endswith(".safetensors") for f in files) or any(
            f.endswith(".bin") for f in files
        ):
            return ModelType.VLLM, None

        return ModelType.UNKNOWN, None

    def _detect_capabilities(self, model_id: str, tags: list[str]) -> dict:
        """Detect model capabilities like thinking and tool use from name and tags."""
        model_lower = model_id.lower()
        tags_lower = [t.lower() for t in tags]

        capabilities = {
            "thinking": False,
            "tool_use": False,
        }

        # Thinking/reasoning model patterns
        thinking_patterns = [
            "thinking", "reason", "-r1", "deepseek-r1", "qwq",
            "o1-", "o3-", "reflection", "cot", "chain-of-thought"
        ]
        if any(p in model_lower for p in thinking_patterns):
            capabilities["thinking"] = True

        # Tool use detection - instruction-tuned models generally support tools
        tool_patterns = [
            "instruct", "chat", "tool", "function", "agent"
        ]
        tool_tags = ["tool-use", "function-calling", "agents"]

        if any(p in model_lower for p in tool_patterns):
            capabilities["tool_use"] = True
        if any(t in tags_lower for t in tool_tags):
            capabilities["tool_use"] = True

        return capabilities

    async def search_models(
        self,
        query: str,
        limit: int = 20,
        sort: str = "downloads",
        model_type: Optional[str] = None,
    ) -> list[dict]:
        """Search for models on HuggingFace.

        Args:
            query: Search query string
            limit: Maximum number of results
            sort: Sort field - "downloads", "likes", "lastModified", "created"
            model_type: Filter by type - "mlx", "gguf", "transformers"
        """

        def _search():
            # Build search query with type filter if specified
            search_query = query
            if model_type:
                if model_type == "mlx":
                    search_query = f"{query} mlx"
                elif model_type == "gguf":
                    search_query = f"{query} gguf"

            # Search for models (sorting is always descending in newer huggingface_hub)
            models = self._hf_api.list_models(
                search=search_query,
                filter="text-generation",
                sort=sort,
                limit=limit,
            )

            results = []
            for m in models:
                # Detect model type from tags
                tags = m.tags if hasattr(m, "tags") else []
                detected_type = "transformers"
                if any("mlx" in t.lower() for t in tags) or "mlx" in m.id.lower():
                    detected_type = "mlx"
                elif any("gguf" in t.lower() for t in tags) or "gguf" in m.id.lower():
                    detected_type = "gguf"

                # Filter by model_type if specified
                if model_type and model_type != detected_type:
                    continue

                # Detect capabilities
                capabilities = self._detect_capabilities(m.id, tags)

                results.append({
                    "id": m.id,
                    "downloads": m.downloads,
                    "likes": m.likes,
                    "tags": tags,
                    "model_type": detected_type,
                    "capabilities": capabilities,
                    "last_modified": m.last_modified.isoformat() if hasattr(m, "last_modified") and m.last_modified else None,
                    "created_at": m.created_at.isoformat() if hasattr(m, "created_at") and m.created_at else None,
                })

            return results

        return await asyncio.to_thread(_search)

    async def register_pending_download(self, repo_id: str, model_id: str) -> ModelInfo:
        """Register a model as downloading before the actual download starts.

        This allows polling to find the model entry immediately.
        """
        local_path = self.models_dir / model_id

        model = ModelInfo(
            id=model_id,
            name=repo_id.split("/")[-1],
            source=repo_id,
            local_path=local_path,
            model_type=ModelType.UNKNOWN,
            status=ModelStatus.DOWNLOADING,
        )

        await self._save_model(model)
        return model

    def _create_progress_tracker(self, model_id: str):
        """Create a tqdm-compatible progress tracker for download progress."""
        manager = self
        import threading

        class ProgressTracker:
            """Custom progress tracker that updates model manager's progress dict."""

            _lock = threading.Lock()

            def __init__(self, iterable=None, *args, **kwargs):
                self.iterable = iterable
                self.total = kwargs.get('total', 0)
                if iterable is not None and self.total == 0:
                    try:
                        self.total = len(iterable)
                    except (TypeError, AttributeError):
                        pass
                self.n = 0
                self.desc = kwargs.get('desc', '')
                self.unit = kwargs.get('unit', 'it')
                self.disable = kwargs.get('disable', False)
                self.pos = kwargs.get('pos', 0)
                self.leave = kwargs.get('leave', True)
                # Store file-level progress
                if model_id not in manager._download_progress:
                    manager._download_progress[model_id] = {
                        'percent': 0.0,
                        'downloaded_bytes': 0,
                        'total_bytes': 0,
                        'current_file': '',
                        'status': 'starting'
                    }
                manager._download_progress[model_id]['current_file'] = self.desc
                manager._download_progress[model_id]['status'] = 'downloading'

            @classmethod
            def get_lock(cls):
                return cls._lock

            @classmethod
            def set_lock(cls, lock):
                cls._lock = lock

            def __iter__(self):
                """Iterate over the wrapped iterable, updating progress."""
                if self.iterable is None:
                    return
                for item in self.iterable:
                    yield item
                    self.update(1)

            def __len__(self):
                return self.total

            def update(self, n=1):
                self.n += n
                if self.total > 0:
                    progress = manager._download_progress[model_id]
                    progress['downloaded_bytes'] += n
                    if progress['total_bytes'] > 0:
                        progress['percent'] = min(99.0, (progress['downloaded_bytes'] / progress['total_bytes']) * 100)

            def close(self):
                pass

            def clear(self):
                pass

            def refresh(self):
                pass

            def reset(self, total=None):
                self.n = 0
                if total is not None:
                    self.total = total

            def display(self, msg=None, pos=None):
                pass

            def set_postfix(self, ordered_dict=None, refresh=True, **kwargs):
                pass

            def set_postfix_str(self, s, refresh=True):
                pass

            def set_description(self, desc=None, refresh=True):
                if desc:
                    self.desc = desc
                    manager._download_progress[model_id]['current_file'] = desc

            def set_description_str(self, desc=None, refresh=True):
                self.set_description(desc, refresh)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.close()

        return ProgressTracker

    async def download_model(
        self,
        repo_id: str,
        revision: str = "main",
    ) -> ModelInfo:
        """Download a model from HuggingFace."""
        # Generate model ID
        model_id = repo_id.replace("/", "--")
        local_path = self.models_dir / model_id

        # Get repo info to detect model type and calculate total size
        def _get_repo_info():
            info = self._hf_api.repo_info(repo_id, files_metadata=True)
            files = [f.rfilename for f in info.siblings]
            total_size = sum(f.size or 0 for f in info.siblings if f.size)
            return files, total_size

        files, total_size = await asyncio.to_thread(_get_repo_info)
        model_type, quantization = self._detect_model_type(repo_id, files)

        # Create model info
        model = ModelInfo(
            id=model_id,
            name=repo_id.split("/")[-1],
            source=repo_id,
            local_path=local_path,
            model_type=model_type,
            quantization=quantization,
            status=ModelStatus.DOWNLOADING,
        )

        # Save to database
        await self._save_model(model)

        # Initialize progress tracking
        self._download_progress[model_id] = {
            'percent': 0.0,
            'downloaded_bytes': 0,
            'total_bytes': total_size,
            'current_file': '',
            'status': 'starting'
        }

        # Download the model
        try:
            def _download():
                # Patch tqdm at multiple levels to ensure proper inline progress display
                # This fixes the issue where progress bars print new lines in threaded contexts
                import functools
                from tqdm.auto import tqdm as original_tqdm

                # Capture these in the closure for the thread context
                _sys = sys
                _os = os

                @functools.wraps(original_tqdm)
                def inline_tqdm(*args, **kwargs):
                    # Force settings that ensure inline updates work in threads
                    if 'file' not in kwargs:
                        kwargs['file'] = _sys.stderr
                    kwargs['leave'] = True
                    kwargs['dynamic_ncols'] = True
                    kwargs['mininterval'] = 0.1
                    # Ensure ncols is set if terminal detection fails in thread
                    if 'ncols' not in kwargs:
                        try:
                            kwargs['ncols'] = _os.get_terminal_size().columns
                        except OSError:
                            kwargs['ncols'] = 100
                    return original_tqdm(*args, **kwargs)

                # Patch tqdm in all places huggingface_hub might import it from
                patches = []
                modules_to_patch = [
                    'huggingface_hub.file_download',
                    'huggingface_hub.utils._tqdm',
                    'huggingface_hub._commit_api',
                ]

                for mod_name in modules_to_patch:
                    try:
                        mod = __import__(mod_name, fromlist=['tqdm'])
                        if hasattr(mod, 'tqdm'):
                            patches.append((mod, 'tqdm', getattr(mod, 'tqdm')))
                            setattr(mod, 'tqdm', inline_tqdm)
                    except (ImportError, AttributeError):
                        pass

                try:
                    return snapshot_download(
                        repo_id,
                        revision=revision,
                        local_dir=str(local_path),
                        tqdm_class=self._create_progress_tracker(model_id),
                    )
                finally:
                    # Restore original tqdm functions
                    for mod, attr, original in patches:
                        setattr(mod, attr, original)

            await asyncio.to_thread(_download)

            model.status = ModelStatus.READY
            self._download_progress[model_id] = {
                'percent': 100.0,
                'downloaded_bytes': total_size,
                'total_bytes': total_size,
                'current_file': '',
                'status': 'complete'
            }
        except Exception as e:
            model.status = ModelStatus.ERROR
            model.metadata["error"] = str(e)
            if model_id in self._download_progress:
                self._download_progress[model_id]['status'] = 'error'
            raise
        finally:
            await self._save_model(model)

        return model

    async def _save_model(self, model: ModelInfo) -> None:
        """Save model info to database."""
        await self._db.execute(
            """
            INSERT INTO models (id, name, source, local_path, model_type, quantization, status, metadata, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                source = excluded.source,
                local_path = excluded.local_path,
                model_type = excluded.model_type,
                quantization = excluded.quantization,
                status = excluded.status,
                metadata = excluded.metadata,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                model.id,
                model.name,
                model.source,
                str(model.local_path) if model.local_path else None,
                model.model_type.value,
                model.quantization,
                model.status.value,
                json.dumps(model.metadata),
            ),
        )
        await self._db.commit()

    async def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """Get a model by ID."""
        async with self._db.execute(
            "SELECT id, name, source, local_path, model_type, quantization, status, metadata FROM models WHERE id = ?",
            (model_id,),
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                return ModelInfo.from_row(row)
        return None

    async def get_model_by_source(self, source: str) -> Optional[ModelInfo]:
        """Get a model by HuggingFace source."""
        async with self._db.execute(
            "SELECT id, name, source, local_path, model_type, quantization, status, metadata FROM models WHERE source = ?",
            (source,),
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                return ModelInfo.from_row(row)
        return None

    async def list_models(self, status: Optional[ModelStatus] = None) -> list[ModelInfo]:
        """List all models."""
        if status:
            query = "SELECT id, name, source, local_path, model_type, quantization, status, metadata FROM models WHERE status = ?"
            params = (status.value,)
        else:
            query = "SELECT id, name, source, local_path, model_type, quantization, status, metadata FROM models"
            params = ()

        models = []
        async with self._db.execute(query, params) as cursor:
            async for row in cursor:
                models.append(ModelInfo.from_row(row))
        return models

    async def delete_model(self, model_id: str) -> bool:
        """Delete a model."""
        model = await self.get_model(model_id)
        if not model:
            return False

        # Delete local files
        if model.local_path and model.local_path.exists():
            import shutil

            shutil.rmtree(model.local_path)

        # Delete from database
        await self._db.execute("DELETE FROM models WHERE id = ?", (model_id,))
        await self._db.commit()
        return True

    def get_download_progress(self, model_id: str) -> Optional[dict]:
        """Get download progress for a model."""
        return self._download_progress.get(model_id)

    def get_all_download_progress(self) -> dict:
        """Get progress for all active downloads."""
        return dict(self._download_progress)

    def _detect_model_type_from_path(self, model_path: Path) -> tuple[ModelType, Optional[str]]:
        """Detect model type and quantization from a local model directory."""
        if not model_path.is_dir():
            return ModelType.UNKNOWN, None

        files = [f.name for f in model_path.iterdir() if f.is_file()]
        dir_name = model_path.name.lower()

        # Check for MLX models
        if "mlx" in dir_name or any("mlx" in f.lower() for f in files):
            quant = None
            if "4bit" in dir_name or "4-bit" in dir_name:
                quant = "4bit"
            elif "8bit" in dir_name or "8-bit" in dir_name:
                quant = "8bit"
            return ModelType.MLX, quant

        # Check for GGUF models
        if any(f.endswith(".gguf") for f in files):
            quant = None
            for q in ["q4_k_m", "q4_0", "q5_k_m", "q5_0", "q8_0", "q6_k"]:
                if q in dir_name:
                    quant = q
                    break
            return ModelType.GGUF, quant

        # Check for quantized models (AWQ, GPTQ)
        if "awq" in dir_name:
            return ModelType.VLLM, "awq"
        if "gptq" in dir_name:
            return ModelType.VLLM, "gptq"

        # Default to vLLM for standard transformers models
        if any(f.endswith(".safetensors") for f in files) or any(f.endswith(".bin") for f in files):
            return ModelType.VLLM, None

        return ModelType.UNKNOWN, None

    def _is_valid_model_dir(self, model_path: Path) -> bool:
        """Check if a directory contains a valid model."""
        if not model_path.is_dir():
            return False

        files = [f.name for f in model_path.iterdir() if f.is_file()]

        # Must have config.json or model weights
        has_config = "config.json" in files
        has_weights = (
            any(f.endswith(".safetensors") for f in files)
            or any(f.endswith(".bin") for f in files)
            or any(f.endswith(".gguf") for f in files)
        )

        return has_config or has_weights

    def _find_model_directories(self, max_depth: int = 2) -> list[Path]:
        """Find all valid model directories up to max_depth levels deep."""
        candidates = []

        if not self.models_dir.exists():
            return candidates

        def scan_level(directory: Path, current_depth: int):
            if current_depth > max_depth:
                return

            try:
                for item in directory.iterdir():
                    if not item.is_dir():
                        continue

                    # Check if this directory contains a valid model
                    if self._is_valid_model_dir(item):
                        candidates.append(item)
                    elif current_depth < max_depth:
                        # Not a model dir, but check subdirectories
                        scan_level(item, current_depth + 1)
            except PermissionError:
                pass  # Skip directories we can't read

        scan_level(self.models_dir, 1)
        return candidates

    async def scan_local_models(self) -> list[ModelInfo]:
        """Scan the models directory for existing models and register them.

        Searches up to 2 levels deep to find models in nested folder structures.
        """
        registered = []

        # Find all valid model directories
        model_dirs = self._find_model_directories(max_depth=2)

        for item in model_dirs:
            model_id = item.name

            # Skip if already registered
            existing = await self.get_model(model_id)
            if existing:
                continue

            # Detect model type
            model_type, quantization = self._detect_model_type_from_path(item)

            # Try to infer source from folder name (owner--model format)
            if "--" in model_id:
                source = model_id.replace("--", "/")
                name = model_id.split("--")[-1]
            else:
                source = f"local/{model_id}"
                name = model_id

            # Create and save model info
            model = ModelInfo(
                id=model_id,
                name=name,
                source=source,
                local_path=item,
                model_type=model_type,
                quantization=quantization,
                status=ModelStatus.READY,
                metadata={"scanned": True},
            )

            await self._save_model(model)
            registered.append(model)

        return registered
