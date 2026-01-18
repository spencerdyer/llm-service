"""OpenAI-compatible API endpoints."""

import json
import time
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from llm_service.backends.base import ChatMessage, CompletionRequest, GenerationConfig
from llm_service.config import settings

router = APIRouter()
security = HTTPBearer()


# --- Pydantic Models ---


class ChatMessageModel(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="default")
    messages: list[ChatMessageModel]
    max_tokens: Optional[int] = Field(default=512, alias="max_tokens")
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    stream: Optional[bool] = False
    stop: Optional[list[str]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    repetition_penalty: Optional[float] = None  # Direct control (1.0 = none, 1.1-1.5 = typical)

    class Config:
        populate_by_name = True


class CompletionRequestModel(BaseModel):
    model: str = Field(default="default")
    prompt: str
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    stream: Optional[bool] = False
    stop: Optional[list[str]] = None
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    repetition_penalty: Optional[float] = None


class UsageModel(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatChoiceModel(BaseModel):
    index: int
    message: ChatMessageModel
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoiceModel]
    usage: UsageModel


class CompletionChoiceModel(BaseModel):
    index: int
    text: str
    finish_reason: str


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoiceModel]
    usage: UsageModel


class ModelObject(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "local"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelObject]


# --- Auth Dependency ---


async def verify_api_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    """Verify the API key."""
    # Get API key from app state (set during startup)
    api_key = getattr(request.app.state, "api_key", None)

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="API key not configured.",
        )

    if credentials.credentials != api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return credentials.credentials


def get_backend_manager(request: Request):
    """Get the backend manager from app state."""
    return request.app.state.backend_manager


def get_model_manager(request: Request):
    """Get the model manager from app state."""
    return request.app.state.model_manager


def get_config_manager(request: Request):
    """Get the config manager from app state."""
    return request.app.state.config_manager


def get_metrics_service(request: Request):
    """Get the metrics service from app state."""
    return getattr(request.app.state, "metrics_service", None)


# --- Endpoints ---


@router.get("/models", response_model=ModelListResponse)
async def list_models(
    request: Request,
    _: str = Depends(verify_api_key),
):
    """List available models."""
    model_manager = get_model_manager(request)
    from llm_service.services.model_manager import ModelStatus

    models = await model_manager.list_models(status=ModelStatus.READY)

    return ModelListResponse(
        data=[
            ModelObject(
                id=m.id,
                created=int(time.time()),
            )
            for m in models
        ]
    )


@router.post("/chat/completions")
async def chat_completions(
    request: Request,
    body: ChatCompletionRequest,
    _: str = Depends(verify_api_key),
):
    """Create a chat completion."""
    backend_manager = get_backend_manager(request)
    config_manager = get_config_manager(request)

    if not backend_manager.is_ready:
        raise HTTPException(status_code=503, detail="No model loaded")

    # Get per-model configuration defaults
    model_id = backend_manager.current_model.id if backend_manager.current_model else None
    model_config = None
    if model_id and config_manager:
        model_config = await config_manager.get_model_config(model_id)

    # Use model config as defaults, allow request to override
    default_temp = model_config.get("temperature", 0.7) if model_config else 0.7
    default_max_tokens = model_config.get("max_tokens", 512) if model_config else 512
    default_top_p = model_config.get("top_p", 0.9) if model_config else 0.9
    default_rep_penalty = model_config.get("repetition_penalty", 1.1) if model_config else 1.1
    default_stop = model_config.get("stop_sequences", []) if model_config else []
    system_prompt = model_config.get("system_prompt") if model_config else None

    # Build messages, prepending system prompt if configured and not already present
    messages = [ChatMessage(role=m.role, content=m.content) for m in body.messages]
    if system_prompt and (not messages or messages[0].role != "system"):
        messages.insert(0, ChatMessage(role="system", content=system_prompt))

    # Calculate repetition penalty
    # If explicitly set, use it; otherwise derive from OpenAI-style penalties or use model default
    if body.repetition_penalty is not None:
        rep_penalty = body.repetition_penalty
    elif (body.frequency_penalty or 0) > 0 or (body.presence_penalty or 0) > 0:
        # Map OpenAI penalties (0-2 range) to repetition penalty (1.0-1.5 range)
        max_penalty = max(body.frequency_penalty or 0, body.presence_penalty or 0)
        rep_penalty = 1.0 + (max_penalty * 0.25)  # 2.0 -> 1.5
    else:
        rep_penalty = default_rep_penalty

    # Use request values if provided, otherwise fall back to model defaults
    config = GenerationConfig(
        max_tokens=body.max_tokens if body.max_tokens is not None else default_max_tokens,
        temperature=body.temperature if body.temperature is not None else default_temp,
        top_p=body.top_p if body.top_p is not None else default_top_p,
        stop=body.stop if body.stop else default_stop,
        repetition_penalty=rep_penalty,
    )
    completion_request = CompletionRequest(
        messages=messages,
        config=config,
        stream=body.stream or False,
    )

    request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())
    model_name = (
        backend_manager.current_model.id if backend_manager.current_model else "unknown"
    )

    if body.stream:
        # Pre-check queue availability for streaming requests
        available, queue_len = backend_manager.check_queue_available()
        if not available:
            raise HTTPException(
                status_code=503,
                detail=f"Request queue is full. {queue_len} requests waiting. Please try again later."
            )

        metrics = get_metrics_service(request)
        # Estimate prompt tokens from message content for streaming
        prompt_text = "".join(m.content for m in body.messages)
        estimated_prompt_tokens = max(1, len(prompt_text) // 4)
        return StreamingResponse(
            _stream_chat_completion(
                backend_manager, metrics, completion_request, request_id, created, model_name,
                estimated_prompt_tokens
            ),
            media_type="text/event-stream",
        )

    # Non-streaming response
    try:
        start_time = time.time()
        response = await backend_manager.generate(completion_request)
        duration = time.time() - start_time
    except RuntimeError as e:
        if "queue" in str(e).lower():
            raise HTTPException(status_code=503, detail=str(e))
        raise HTTPException(status_code=500, detail=str(e))

    # Record metrics
    metrics = get_metrics_service(request)
    if metrics:
        tokens_per_second = response.completion_tokens / duration if duration > 0 else 0
        await metrics.record_request(
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            tokens_per_second=tokens_per_second,
        )

    return ChatCompletionResponse(
        id=request_id,
        created=created,
        model=model_name,
        choices=[
            ChatChoiceModel(
                index=0,
                message=ChatMessageModel(role="assistant", content=response.text),
                finish_reason=response.finish_reason,
            )
        ],
        usage=UsageModel(
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
        ),
    )


async def _stream_chat_completion(
    backend_manager, metrics_service, request: CompletionRequest, request_id: str, created: int, model: str,
    estimated_prompt_tokens: int = 0
):
    """Stream chat completion chunks."""
    total_text = ""
    start_time = time.time()
    try:
        async for chunk in backend_manager.generate_stream(request):
            if chunk.text:
                total_text += chunk.text
            data = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": chunk.text} if chunk.text else {},
                        "finish_reason": chunk.finish_reason,
                    }
                ],
            }
            yield f"data: {json.dumps(data)}\n\n"

        yield "data: [DONE]\n\n"

        # Record metrics
        if metrics_service:
            estimated_completion_tokens = max(1, len(total_text) // 4)
            duration = time.time() - start_time
            tokens_per_second = estimated_completion_tokens / duration if duration > 0 else 0
            await metrics_service.record_request(
                prompt_tokens=estimated_prompt_tokens,
                completion_tokens=estimated_completion_tokens,
                tokens_per_second=tokens_per_second,
            )
    except Exception as e:
        error_data = {"error": {"message": str(e), "type": "server_error"}}
        yield f"data: {json.dumps(error_data)}\n\n"


@router.post("/completions")
async def completions(
    request: Request,
    body: CompletionRequestModel,
    _: str = Depends(verify_api_key),
):
    """Create a completion."""
    backend_manager = get_backend_manager(request)
    config_manager = get_config_manager(request)

    if not backend_manager.is_ready:
        raise HTTPException(status_code=503, detail="No model loaded")

    # Get per-model configuration defaults
    model_id = backend_manager.current_model.id if backend_manager.current_model else None
    model_config = None
    if model_id and config_manager:
        model_config = await config_manager.get_model_config(model_id)

    # Use model config as defaults
    default_temp = model_config.get("temperature", 0.7) if model_config else 0.7
    default_max_tokens = model_config.get("max_tokens", 512) if model_config else 512
    default_top_p = model_config.get("top_p", 0.9) if model_config else 0.9
    default_rep_penalty = model_config.get("repetition_penalty", 1.1) if model_config else 1.1
    default_stop = model_config.get("stop_sequences", []) if model_config else []

    # Calculate repetition penalty
    if body.repetition_penalty is not None:
        rep_penalty = body.repetition_penalty
    elif (body.frequency_penalty or 0) > 0 or (body.presence_penalty or 0) > 0:
        max_penalty = max(body.frequency_penalty or 0, body.presence_penalty or 0)
        rep_penalty = 1.0 + (max_penalty * 0.25)
    else:
        rep_penalty = default_rep_penalty

    config = GenerationConfig(
        max_tokens=body.max_tokens if body.max_tokens is not None else default_max_tokens,
        temperature=body.temperature if body.temperature is not None else default_temp,
        top_p=body.top_p if body.top_p is not None else default_top_p,
        stop=body.stop if body.stop else default_stop,
        repetition_penalty=rep_penalty,
    )
    completion_request = CompletionRequest(
        prompt=body.prompt,
        config=config,
        stream=body.stream or False,
    )

    request_id = f"cmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())
    model_name = (
        backend_manager.current_model.id if backend_manager.current_model else "unknown"
    )

    if body.stream:
        # Pre-check queue availability for streaming requests
        available, queue_len = backend_manager.check_queue_available()
        if not available:
            raise HTTPException(
                status_code=503,
                detail=f"Request queue is full. {queue_len} requests waiting. Please try again later."
            )

        metrics = get_metrics_service(request)
        # Estimate prompt tokens from prompt text for streaming
        estimated_prompt_tokens = max(1, len(body.prompt) // 4)
        return StreamingResponse(
            _stream_completion(
                backend_manager, metrics, completion_request, request_id, created, model_name,
                estimated_prompt_tokens
            ),
            media_type="text/event-stream",
        )

    try:
        start_time = time.time()
        response = await backend_manager.generate(completion_request)
        duration = time.time() - start_time
    except RuntimeError as e:
        if "queue" in str(e).lower():
            raise HTTPException(status_code=503, detail=str(e))
        raise HTTPException(status_code=500, detail=str(e))

    # Record metrics
    metrics = get_metrics_service(request)
    if metrics:
        tokens_per_second = response.completion_tokens / duration if duration > 0 else 0
        await metrics.record_request(
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            tokens_per_second=tokens_per_second,
        )

    return CompletionResponse(
        id=request_id,
        created=created,
        model=model_name,
        choices=[
            CompletionChoiceModel(
                index=0,
                text=response.text,
                finish_reason=response.finish_reason,
            )
        ],
        usage=UsageModel(
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
        ),
    )


async def _stream_completion(
    backend_manager, metrics_service, request: CompletionRequest, request_id: str, created: int, model: str,
    estimated_prompt_tokens: int = 0
):
    """Stream completion chunks."""
    total_text = ""
    start_time = time.time()
    try:
        async for chunk in backend_manager.generate_stream(request):
            if chunk.text:
                total_text += chunk.text
            data = {
                "id": request_id,
                "object": "text_completion",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "text": chunk.text,
                        "finish_reason": chunk.finish_reason,
                    }
                ],
            }
            yield f"data: {json.dumps(data)}\n\n"

        yield "data: [DONE]\n\n"

        # Record metrics
        if metrics_service:
            estimated_completion_tokens = max(1, len(total_text) // 4)
            duration = time.time() - start_time
            tokens_per_second = estimated_completion_tokens / duration if duration > 0 else 0
            await metrics_service.record_request(
                prompt_tokens=estimated_prompt_tokens,
                completion_tokens=estimated_completion_tokens,
                tokens_per_second=tokens_per_second,
            )
    except Exception as e:
        error_data = {"error": {"message": str(e), "type": "server_error"}}
        yield f"data: {json.dumps(error_data)}\n\n"
