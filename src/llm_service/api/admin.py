"""Admin API and UI endpoints."""

import asyncio
import json
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from llm_service.config import settings
from llm_service.services.model_manager import ModelStatus

router = APIRouter()
api_router = APIRouter()


# --- Pydantic Models ---


class DownloadModelRequest(BaseModel):
    repo_id: str
    revision: str = "main"


class LoadModelRequest(BaseModel):
    model_id: str


class SettingUpdate(BaseModel):
    key: str
    value: str


class SearchModelsRequest(BaseModel):
    query: str
    limit: int = 20
    sort: str = "downloads"  # downloads, likes, lastModified, created
    model_type: Optional[str] = None  # mlx, gguf, transformers


class SetApiKeyRequest(BaseModel):
    api_key: str


class SetModelsDirRequest(BaseModel):
    path: str


class UnloadModelRequest(BaseModel):
    model_id: str


class SetActiveModelRequest(BaseModel):
    model_id: str


class ModelConfigRequest(BaseModel):
    display_name: Optional[str] = None
    system_prompt: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    context_length: int = 4096
    stop_sequences: list[str] = []


# --- Helper Functions ---


def get_backend_manager(request: Request):
    return request.app.state.backend_manager


def get_model_manager(request: Request):
    return request.app.state.model_manager


def get_config_manager(request: Request):
    return request.app.state.config_manager


# --- Admin UI Routes ---


@router.get("/", response_class=HTMLResponse)
async def admin_dashboard(request: Request):
    """Render the admin dashboard."""
    templates = request.app.state.templates
    backend_manager = get_backend_manager(request)
    model_manager = get_model_manager(request)
    config_manager = get_config_manager(request)

    models = await model_manager.list_models()
    status = backend_manager.get_status()
    all_settings = await config_manager.get_all_settings()

    # Get current API key
    current_api_key = getattr(request.app.state, "api_key", None)

    return templates.TemplateResponse(
        "admin/dashboard.html",
        {
            "request": request,
            "status": status,
            "models": [m.to_dict() for m in models],
            "settings": all_settings,
            "platform": settings.platform.value,
            "api_key": current_api_key,
            "models_dir": str(model_manager.models_dir),
        },
    )


# --- Admin API Routes ---


@api_router.get("/status")
async def get_status(request: Request):
    """Get current service status."""
    backend_manager = get_backend_manager(request)
    return backend_manager.get_status()


@api_router.get("/models")
async def list_models(request: Request, status: Optional[str] = None):
    """List all models."""
    model_manager = get_model_manager(request)

    model_status = None
    if status:
        try:
            model_status = ModelStatus(status)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}")

    models = await model_manager.list_models(status=model_status)
    return {"models": [m.to_dict() for m in models]}


# NOTE: These download progress endpoints must come BEFORE /models/{model_id}
# to avoid "downloads" being treated as a model_id
@api_router.get("/models/downloads")
async def get_download_progress(request: Request):
    """Get progress for all active downloads."""
    model_manager = get_model_manager(request)
    return {"downloads": model_manager.get_all_download_progress()}


@api_router.get("/models/downloads/{model_id}")
async def get_model_download_progress(request: Request, model_id: str):
    """Get download progress for a specific model."""
    model_manager = get_model_manager(request)
    progress = model_manager.get_download_progress(model_id)

    if progress is None:
        # Check if model exists and is complete
        model = await model_manager.get_model(model_id)
        if model:
            if model.status == ModelStatus.READY:
                return {"status": "complete", "percent": 100.0}
            elif model.status == ModelStatus.ERROR:
                return {"status": "error", "percent": 0.0}
        raise HTTPException(status_code=404, detail="No active download for this model")

    return progress


@api_router.get("/models/downloads/{model_id}/stream")
async def stream_download_progress(request: Request, model_id: str):
    """Stream download progress for a specific model using Server-Sent Events."""
    model_manager = get_model_manager(request)

    async def event_generator():
        while True:
            # Check if client disconnected
            if await request.is_disconnected():
                break

            progress = model_manager.get_download_progress(model_id)

            if progress is None:
                # Check if model exists and is complete
                model = await model_manager.get_model(model_id)
                if model:
                    if model.status == ModelStatus.READY:
                        yield f"data: {json.dumps({'status': 'complete', 'percent': 100.0})}\n\n"
                        break
                    elif model.status == ModelStatus.ERROR:
                        yield f"data: {json.dumps({'status': 'error', 'percent': 0.0})}\n\n"
                        break
                # No progress yet, keep waiting
                yield f"data: {json.dumps({'status': 'waiting', 'percent': 0.0})}\n\n"
            else:
                yield f"data: {json.dumps(progress)}\n\n"
                if progress.get("status") == "complete":
                    break

            await asyncio.sleep(0.5)  # Update every 500ms

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@api_router.get("/models/{model_id}")
async def get_model(request: Request, model_id: str):
    """Get a specific model."""
    model_manager = get_model_manager(request)
    model = await model_manager.get_model(model_id)

    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    return model.to_dict()


@api_router.post("/models/{model_id}/favorite")
async def toggle_model_favorite(request: Request, model_id: str):
    """Toggle favorite status for a model."""
    model_manager = get_model_manager(request)
    model = await model_manager.get_model(model_id)

    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # Toggle favorite in metadata
    is_favorite = model.metadata.get("favorite", False)
    model.metadata["favorite"] = not is_favorite
    await model_manager._save_model(model)

    return {"status": "updated", "favorite": model.metadata["favorite"]}


@api_router.post("/models/search")
async def search_models(request: Request, body: SearchModelsRequest):
    """Search for models on HuggingFace."""
    model_manager = get_model_manager(request)

    try:
        results = await model_manager.search_models(
            body.query,
            limit=body.limit,
            sort=body.sort,
            model_type=body.model_type,
        )
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/models/download")
async def download_model(
    request: Request, body: DownloadModelRequest, background_tasks: BackgroundTasks
):
    """Start downloading a model."""
    model_manager = get_model_manager(request)

    # Check if already downloaded or downloading
    existing = await model_manager.get_model_by_source(body.repo_id)
    if existing:
        if existing.status == ModelStatus.READY:
            return {"status": "already_downloaded", "model": existing.to_dict()}
        elif existing.status == ModelStatus.DOWNLOADING:
            return {"status": "downloading", "message": f"Already downloading {body.repo_id}"}

    # Register model immediately with DOWNLOADING status so polling can find it
    model_id = body.repo_id.replace("/", "--")
    await model_manager.register_pending_download(body.repo_id, model_id)

    # Start download in background using asyncio.run in a sync wrapper
    def run_download():
        asyncio.run(do_download_async())

    async def do_download_async():
        try:
            await model_manager.download_model(body.repo_id, revision=body.revision)
        except Exception as e:
            print(f"Download failed: {e}")

    background_tasks.add_task(run_download)

    return {
        "status": "downloading",
        "message": f"Started downloading {body.repo_id}",
        "model_id": model_id,
    }


@api_router.post("/models/scan")
async def scan_local_models(request: Request):
    """Scan the models directory for existing models and register them."""
    model_manager = get_model_manager(request)

    try:
        registered = await model_manager.scan_local_models()
        return {
            "status": "completed",
            "registered": len(registered),
            "models": [m.to_dict() for m in registered],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/models-dir")
async def get_models_dir(request: Request):
    """Get the current models directory."""
    model_manager = get_model_manager(request)
    return {"path": str(model_manager.models_dir)}


@api_router.post("/models-dir")
async def set_models_dir(request: Request, body: SetModelsDirRequest):
    """Set the models directory."""
    model_manager = get_model_manager(request)

    from pathlib import Path

    path = Path(body.path).expanduser()

    # Validate the path
    if not path.is_absolute():
        raise HTTPException(status_code=400, detail="Path must be absolute")

    try:
        await model_manager.set_models_dir(body.path)
        return {"status": "updated", "path": str(model_manager.models_dir)}
    except PermissionError:
        raise HTTPException(status_code=400, detail="Permission denied for this path")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api_router.delete("/models/{model_id}")
async def delete_model(request: Request, model_id: str):
    """Delete a model."""
    model_manager = get_model_manager(request)
    backend_manager = get_backend_manager(request)

    # Unload if currently loaded
    await backend_manager.unload_model(model_id)

    success = await model_manager.delete_model(model_id)
    if not success:
        raise HTTPException(status_code=404, detail="Model not found")

    return {"status": "deleted"}


@api_router.post("/models/load")
async def load_model(request: Request, body: LoadModelRequest):
    """Load a model for inference."""
    backend_manager = get_backend_manager(request)

    try:
        model = await backend_manager.load_model(body.model_id)
        return {
            "status": "loaded",
            "model": {
                "id": model.id,
                "name": model.name,
                "source": model.source,
            },
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/models/unload")
async def unload_model(request: Request, body: UnloadModelRequest = None):
    """Unload a model. If no model_id provided, unload the active model."""
    backend_manager = get_backend_manager(request)
    model_id = body.model_id if body else None
    await backend_manager.unload_model(model_id)
    return {"status": "unloaded"}


@api_router.post("/models/unload-all")
async def unload_all_models(request: Request):
    """Unload all loaded models."""
    backend_manager = get_backend_manager(request)
    await backend_manager.unload_all_models()
    return {"status": "unloaded_all"}


@api_router.post("/models/set-active")
async def set_active_model(request: Request, body: SetActiveModelRequest):
    """Set the active model for generation."""
    backend_manager = get_backend_manager(request)

    if backend_manager.set_active_model(body.model_id):
        return {"status": "active", "model_id": body.model_id}
    else:
        raise HTTPException(status_code=400, detail="Model not loaded")


@api_router.get("/models/loaded")
async def get_loaded_models(request: Request):
    """Get all currently loaded models."""
    backend_manager = get_backend_manager(request)
    return {
        "models": backend_manager.get_loaded_models_info(),
        "total_memory_mb": backend_manager.get_total_memory_mb(),
    }


# --- Model Configuration ---


@api_router.get("/models/{model_id}/config")
async def get_model_config(request: Request, model_id: str):
    """Get configuration for a specific model."""
    config_manager = get_config_manager(request)
    model_manager = get_model_manager(request)

    # Check model exists
    model = await model_manager.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    config = await config_manager.get_model_config(model_id)
    if config:
        return config

    # Return defaults if no config exists
    return {
        "model_id": model_id,
        "display_name": model.name,
        "system_prompt": None,
        "temperature": 0.7,
        "max_tokens": 2048,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "context_length": 4096,
        "stop_sequences": [],
    }


@api_router.put("/models/{model_id}/config")
async def update_model_config(request: Request, model_id: str, body: ModelConfigRequest):
    """Update configuration for a specific model."""
    config_manager = get_config_manager(request)
    model_manager = get_model_manager(request)

    # Check model exists
    model = await model_manager.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    await config_manager.set_model_config(model_id, body.model_dump())
    return {"status": "updated", "model_id": model_id}


@api_router.delete("/models/{model_id}/config")
async def delete_model_config(request: Request, model_id: str):
    """Delete configuration for a specific model (resets to defaults)."""
    config_manager = get_config_manager(request)
    await config_manager.delete_model_config(model_id)
    return {"status": "deleted", "model_id": model_id}


@api_router.get("/settings")
async def get_settings(request: Request):
    """Get all settings."""
    config_manager = get_config_manager(request)
    return await config_manager.get_all_settings()


@api_router.post("/settings")
async def update_setting(request: Request, body: SettingUpdate):
    """Update a setting."""
    config_manager = get_config_manager(request)
    await config_manager.set_setting(body.key, body.value)
    return {"status": "updated", "key": body.key}


@api_router.delete("/settings/{key}")
async def delete_setting(request: Request, key: str):
    """Delete a setting."""
    config_manager = get_config_manager(request)
    await config_manager.delete_setting(key)
    return {"status": "deleted", "key": key}


# --- API Key Management ---


@api_router.get("/api-key")
async def get_api_key(request: Request):
    """Get the current API key."""
    api_key = getattr(request.app.state, "api_key", None)
    return {"api_key": api_key}


@api_router.post("/api-key")
async def set_api_key(request: Request, body: SetApiKeyRequest):
    """Set a new API key."""
    config_manager = get_config_manager(request)

    if not body.api_key or len(body.api_key) < 8:
        raise HTTPException(status_code=400, detail="API key must be at least 8 characters")

    # Update in config manager (persists to DB)
    await config_manager.set_setting("api_key", body.api_key)

    # Update runtime state
    request.app.state.api_key = body.api_key

    return {"status": "updated", "api_key": body.api_key}


@api_router.post("/api-key/regenerate")
async def regenerate_api_key(request: Request):
    """Generate a new random API key."""
    from llm_service.config import generate_api_key

    config_manager = get_config_manager(request)

    # Generate new key
    new_key = generate_api_key()

    # Update in config manager (persists to DB)
    await config_manager.set_setting("api_key", new_key)

    # Update runtime state
    request.app.state.api_key = new_key

    return {"status": "regenerated", "api_key": new_key}


# --- Metrics ---


def get_metrics_service(request: Request):
    return getattr(request.app.state, "metrics_service", None)


@api_router.get("/metrics/realtime")
async def get_realtime_metrics(request: Request):
    """Get per-second metrics for the last 60 seconds."""
    metrics = get_metrics_service(request)
    if not metrics:
        raise HTTPException(status_code=503, detail="Metrics service not available")
    return {"data": await metrics.get_realtime_metrics()}


@api_router.get("/metrics/historical")
async def get_historical_metrics(request: Request, minutes: int = 60):
    """Get per-minute metrics for the specified time window."""
    metrics = get_metrics_service(request)
    if not metrics:
        raise HTTPException(status_code=503, detail="Metrics service not available")
    if minutes < 1 or minutes > 1440:
        raise HTTPException(status_code=400, detail="Minutes must be between 1 and 1440")
    return {"data": await metrics.get_historical_metrics(minutes)}


@api_router.get("/metrics/summary")
async def get_metrics_summary(request: Request):
    """Get summary statistics."""
    metrics = get_metrics_service(request)
    if not metrics:
        raise HTTPException(status_code=503, detail="Metrics service not available")
    return await metrics.get_summary()


@api_router.get("/metrics/stream")
async def stream_metrics(request: Request):
    """Stream real-time metrics using Server-Sent Events."""
    metrics = get_metrics_service(request)
    if not metrics:
        raise HTTPException(status_code=503, detail="Metrics service not available")

    async def event_generator():
        while True:
            if await request.is_disconnected():
                break
            data = await metrics.get_realtime_metrics()
            # Only send the last data point for efficiency
            latest = data[-1] if data else {"timestamp": 0, "requests": 0, "prompt_tokens": 0, "completion_tokens": 0}
            yield f"data: {json.dumps(latest)}\n\n"
            await asyncio.sleep(1)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
