"""OpenAI-compatible API endpoints."""

import json
import time
import uuid
from typing import Any, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, field_validator

from llm_service.backends.base import (
    ChatMessage,
    CompletionRequest,
    split_thinking_content,
)
from llm_service.backends.multimodal import normalize_content, text_from_content
from llm_service.backends.sampling import (
    apply_system_prompts,
    build_generation_config,
)
from llm_service.backends.tool_parse import (
    inject_tool_constraint_message,
    normalize_tool_choice,
    tool_call_name,
    tool_names_from_tools,
    tools_are_active,
)
from llm_service.services.metrics import explain_generation_end

router = APIRouter()
security = HTTPBearer()


# --- Pydantic Models ---


class ToolCallFunctionModel(BaseModel):
    name: Optional[str] = None
    arguments: Optional[str] = None


class ToolCallModel(BaseModel):
    id: Optional[str] = None
    type: str = "function"
    function: ToolCallFunctionModel
    index: Optional[int] = None


class ChatMessageModel(BaseModel):
    role: str
    content: Optional[Any] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[list[ToolCallModel]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

    @field_validator("content", mode="before")
    @classmethod
    def flatten_content(cls, value):
        """Keep image/video parts; flatten text-only arrays to a string."""
        return normalize_content(value)


class StreamOptionsModel(BaseModel):
    include_usage: Optional[bool] = False


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="default")
    messages: list[ChatMessageModel]
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptionsModel] = None
    stop: Optional[Union[str, list[str]]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    repetition_penalty: Optional[float] = None  # Direct control (1.0 = none, 1.1-1.5 = typical)
    # Hybrid thinking models (e.g. Qwen3/3.5/3.6): force thinking on/off via chat template
    enable_thinking: Optional[bool] = None
    chat_template_kwargs: Optional[dict] = None
    tools: Optional[list[dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    system: Optional[str] = None
    sampling_profile: Optional[str] = None  # auto | text | tools | request

    class Config:
        populate_by_name = True
        extra = "ignore"

    @field_validator("stop", mode="before")
    @classmethod
    def normalize_stop(cls, value):
        if value is None or value == "":
            return None
        if isinstance(value, str):
            return [value]
        return value


class CompletionRequestModel(BaseModel):
    model: str = Field(default="default")
    prompt: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptionsModel] = None
    stop: Optional[Union[str, list[str]]] = None
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    repetition_penalty: Optional[float] = None
    enable_thinking: Optional[bool] = None
    chat_template_kwargs: Optional[dict] = None
    max_completion_tokens: Optional[int] = None


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


def ensure_inference_mode(request: Request) -> None:
    """Block normal inference while the app is in workbench mode."""
    operation_mode = getattr(request.app.state, "operation_mode", "inference")
    if operation_mode != "inference":
        raise HTTPException(
            status_code=503,
            detail="The app is in workbench mode. Switch back to inference mode to use /v1 endpoints.",
        )


# --- Endpoints ---


def _resolve_loaded_model(backend_manager, model_ref: Optional[str]):
    """Resolve a client model id to a loaded model, or raise an HTTP error."""
    try:
        return backend_manager.resolve_request_model(model_ref)
    except ValueError as e:
        message = str(e)
        status = 503 if message.startswith("No model loaded") else 404
        raise HTTPException(status_code=status, detail=message) from e


def _resolve_max_tokens(
    max_tokens: Optional[int],
    max_completion_tokens: Optional[int],
    default_max_tokens: int,
) -> int:
    """Prefer the request's max_completion_tokens (Grok / new OpenAI), then max_tokens."""
    from llm_service.backends.sampling import resolve_max_tokens

    return resolve_max_tokens(max_tokens, max_completion_tokens, default_max_tokens)


def _to_chat_messages(body_messages: list[ChatMessageModel]) -> list[ChatMessage]:
    messages: list[ChatMessage] = []
    for item in body_messages:
        tool_calls = None
        if item.tool_calls:
            tool_calls = [tc.model_dump(exclude_none=True) for tc in item.tool_calls]
        messages.append(
            ChatMessage(
                role=item.role,
                content=item.content,
                tool_calls=tool_calls,
                tool_call_id=item.tool_call_id,
                name=item.name,
            )
        )
    return messages


def _resolve_enable_thinking(
    enable_thinking: Optional[bool],
    chat_template_kwargs: Optional[dict],
) -> Optional[bool]:
    """Resolve thinking mode from top-level field or chat_template_kwargs."""
    if enable_thinking is not None:
        return enable_thinking
    if chat_template_kwargs and "enable_thinking" in chat_template_kwargs:
        return bool(chat_template_kwargs["enable_thinking"])
    return None


def _resolve_temperature(
    requested: Optional[float],
    default_temp: float,
    tools: Optional[list] = None,
    tool_choice: Any = None,
    sampling_profile: Optional[str] = None,
) -> float:
    """Resolve temperature using text vs tool sampling profiles."""
    _, config = build_generation_config(
        model_config={"temperature": default_temp},
        sampling_profile=sampling_profile,
        tools=tools,
        tool_choice=tool_choice,
        temperature=requested,
    )
    return config.temperature


def _preview_text(value: Any, limit: int = 400) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        value = text_from_content(value)
    text = " ".join(str(value).split())
    if len(text) > limit:
        return text[: limit - 1] + "…"
    return text


def _user_preview(messages: list[ChatMessage]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            return _preview_text(message.content)
    return ""


def _serialize_messages(
    messages: Optional[list[ChatMessage]],
) -> tuple[Optional[list[dict[str, Any]]], Optional[str]]:
    if not messages:
        return None, None
    payload: list[dict[str, Any]] = []
    lines: list[str] = []
    for message in messages:
        item: dict[str, Any] = {"role": message.role, "content": message.content}
        if message.tool_calls:
            item["tool_calls"] = message.tool_calls
        if message.tool_call_id:
            item["tool_call_id"] = message.tool_call_id
        if message.name:
            item["name"] = message.name
        payload.append(item)
        content = message.content
        if isinstance(content, list):
            content = text_from_content(content)
        block = f"[{message.role}]\n{'' if content is None else content}"
        if message.tool_calls:
            block += "\n" + json.dumps(message.tool_calls, ensure_ascii=False, default=str)
        if message.tool_call_id:
            block += f"\n(tool_call_id={message.tool_call_id})"
        lines.append(block)
    return payload, "\n\n".join(lines)


def _sampling_payload(request: CompletionRequest) -> dict[str, Any]:
    cfg = request.config
    return {
        "max_tokens": cfg.max_tokens,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "top_k": cfg.top_k,
        "repetition_penalty": cfg.repetition_penalty,
        "stop": list(cfg.stop or []),
        "enable_thinking": request.enable_thinking,
        "stream": request.stream,
        "tool_choice": request.tool_choice,
        "tool_count": len(request.tools or []),
        "tool_names": tool_names_from_tools(request.tools),
    }


def _call_body_fields(
    request: CompletionRequest,
    *,
    status: str,
    finish_reason: Optional[str],
    engine_finish_reason: Optional[str] = None,
    completion_text: str = "",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    token_count_source: str = "estimated",
    tool_calls: Optional[list] = None,
    error: Optional[str] = None,
) -> dict[str, Any]:
    """Full prompt/response bodies plus a diagnosis of why generation ended."""
    messages, transcript = _serialize_messages(request.messages)
    if not transcript and request.prompt:
        transcript = request.prompt
    reasoning, answer = split_thinking_content(completion_text or "")
    explanation, truncated = explain_generation_end(
        status=status,
        finish_reason=finish_reason,
        engine_finish_reason=engine_finish_reason,
        max_tokens=request.config.max_tokens,
        completion_tokens=completion_tokens,
        token_count_source=token_count_source,
        stop=request.config.stop,
        error=error,
        reasoning_chars=len(reasoning or ""),
        answer_chars=len(answer or ""),
    )
    return {
        "prompt_preview": _user_preview(request.messages or [])
        or _preview_text(request.prompt),
        "completion_preview": _preview_text(answer or completion_text),
        "prompt_text": transcript,
        "completion_text": completion_text,
        "reasoning_text": reasoning,
        "messages": messages,
        "sampling": _sampling_payload(request),
        "stop_explanation": explanation,
        "max_tokens": request.config.max_tokens,
        "engine_finish_reason": engine_finish_reason,
        "token_count_source": token_count_source,
        "truncated": truncated,
        "tool_calls": tool_calls,
        "error": error,
    }


def _tool_choice_label(tool_choice: Any) -> Optional[str]:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        return tool_choice
    kind, named = normalize_tool_choice(tool_choice)
    if named:
        return named
    return kind


def _names_from_tool_calls(tool_calls: Optional[list]) -> list[str]:
    names: list[str] = []
    for call in tool_calls or []:
        if hasattr(call, "function"):
            name = getattr(call.function, "name", None)
        elif isinstance(call, dict):
            name = tool_call_name(call)
            if not name and isinstance(call.get("function"), dict):
                name = call["function"].get("name")
        else:
            name = None
        if name:
            names.append(str(name))
    return names


async def _log_call(metrics, **kwargs) -> None:
    if not metrics:
        return
    try:
        await metrics.record_request(
            prompt_tokens=kwargs.pop("prompt_tokens", 0),
            completion_tokens=kwargs.pop("completion_tokens", 0),
            tokens_per_second=kwargs.pop("tokens_per_second", 0.0),
            **kwargs,
        )
    except Exception:
        pass


@router.get("/models", response_model=ModelListResponse)
async def list_models(
    request: Request,
    _: str = Depends(verify_api_key),
):
    """List models currently loaded and available for inference."""
    ensure_inference_mode(request)
    backend_manager = get_backend_manager(request)
    models = backend_manager.loaded_models

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
    ensure_inference_mode(request)
    backend_manager = get_backend_manager(request)
    config_manager = get_config_manager(request)

    selected_model = _resolve_loaded_model(backend_manager, body.model)
    model_id = selected_model.id

    # Get per-model configuration defaults
    model_config = None
    if config_manager:
        model_config = await config_manager.get_model_config(model_id)

    using_tools = tools_are_active(body.tools, body.tool_choice)
    messages = apply_system_prompts(
        _to_chat_messages(body.messages),
        stored_system=(model_config or {}).get("system_prompt"),
        request_system=body.system,
        tool_system=(model_config or {}).get("tool_system_prompt"),
        tools_active=using_tools,
    )
    messages = inject_tool_constraint_message(messages, body.tools, body.tool_choice)

    _, config = build_generation_config(
        model_config=model_config,
        sampling_profile=body.sampling_profile,
        tools=body.tools,
        tool_choice=body.tool_choice,
        temperature=body.temperature,
        top_p=body.top_p,
        top_k=body.top_k,
        repetition_penalty=body.repetition_penalty,
        frequency_penalty=body.frequency_penalty,
        presence_penalty=body.presence_penalty,
        max_tokens=body.max_tokens,
        max_completion_tokens=body.max_completion_tokens,
        stop=body.stop,
    )
    completion_request = CompletionRequest(
        messages=messages,
        config=config,
        stream=body.stream or False,
        enable_thinking=_resolve_enable_thinking(
            body.enable_thinking, body.chat_template_kwargs
        ),
        tools=body.tools or None,
        tool_choice=body.tool_choice,
    )

    request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())
    model_name = model_id

    if body.stream:
        # Pre-check queue availability for streaming requests
        available, queue_len = backend_manager.check_queue_available(model_id)
        if not available:
            raise HTTPException(
                status_code=503,
                detail=f"Request queue is full. {queue_len} requests waiting. Please try again later."
            )

        metrics = get_metrics_service(request)
        include_usage = (
            body.stream_options.include_usage
            if body.stream_options and body.stream_options.include_usage
            else False
        )
        # Estimate prompt tokens from message content for streaming
        prompt_text = "".join(text_from_content(m.content) for m in body.messages)
        estimated_prompt_tokens = max(1, len(prompt_text) // 4)
        return StreamingResponse(
            _stream_chat_completion(
                backend_manager, metrics, completion_request, request_id, created, model_name,
                estimated_prompt_tokens, include_usage, model_id=model_id,
            ),
            media_type="text/event-stream",
        )

    # Non-streaming response
    metrics = get_metrics_service(request)
    try:
        start_time = time.time()
        response = await backend_manager.generate(completion_request, model_id=model_id)
        duration = time.time() - start_time
    except RuntimeError as e:
        await _log_call(
            metrics,
            prompt_tokens=0,
            completion_tokens=0,
            request_id=request_id,
            model=model_name,
            endpoint="chat.completions",
            stream=False,
            status="error",
            has_tools=tools_are_active(body.tools, body.tool_choice),
            tool_choice=_tool_choice_label(body.tool_choice),
            **_call_body_fields(
                completion_request,
                status="error",
                finish_reason="error",
                completion_text="",
                error=str(e),
            ),
        )
        if "queue" in str(e).lower():
            raise HTTPException(status_code=503, detail=str(e))
        raise HTTPException(status_code=500, detail=str(e))

    reasoning, answer = split_thinking_content(response.text or "")
    tool_calls = None
    if response.tool_calls:
        tool_calls = [ToolCallModel.model_validate(tc) for tc in response.tool_calls]

    tokens_per_second = response.completion_tokens / duration if duration > 0 else 0
    await _log_call(
        metrics,
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        tokens_per_second=tokens_per_second,
        request_id=request_id,
        model=model_name,
        endpoint="chat.completions",
        stream=False,
        status="ok",
        duration_ms=round(duration * 1000, 1),
        finish_reason=response.finish_reason,
        has_tools=tools_are_active(body.tools, body.tool_choice),
        tool_choice=_tool_choice_label(body.tool_choice),
        tool_names=_names_from_tool_calls(response.tool_calls),
        rejected_tool_names=response.rejected_tool_names or [],
        **_call_body_fields(
            completion_request,
            status="ok",
            finish_reason=response.finish_reason,
            engine_finish_reason=response.engine_finish_reason or response.finish_reason,
            completion_text=response.text or "",
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            token_count_source="engine",
            tool_calls=response.tool_calls,
        ),
    )

    return ChatCompletionResponse(
        id=request_id,
        created=created,
        model=model_name,
        choices=[
            ChatChoiceModel(
                index=0,
                message=ChatMessageModel(
                    role="assistant",
                    content=answer or (None if tool_calls else ""),
                    reasoning_content=reasoning,
                    tool_calls=tool_calls,
                ),
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
    estimated_prompt_tokens: int = 0, include_usage: bool = False, model_id: Optional[str] = None,
):
    """Stream chat completion chunks per OpenAI spec."""
    import asyncio

    total_text = ""
    start_time = time.time()
    finish_reason = None
    tool_names: list[str] = []
    rejected_tool_names: list[str] = []
    streamed_tool_calls: list = []
    status = "ok"
    error = None
    engine_prompt_tokens: Optional[int] = None
    engine_completion_tokens: Optional[int] = None
    engine_finish_reason: Optional[str] = None

    try:
        # Initial chunk: signal role to the client
        initial_data = json.dumps({
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": None,
            }],
        })
        yield f"data: {initial_data}\n\n"

        # Content / tool-call chunks
        async for chunk in backend_manager.generate_stream(request, model_id=model_id):
            if chunk.text:
                total_text += chunk.text
            if chunk.finish_reason:
                finish_reason = chunk.finish_reason
            if chunk.engine_finish_reason:
                engine_finish_reason = chunk.engine_finish_reason
            if chunk.prompt_tokens:
                engine_prompt_tokens = chunk.prompt_tokens
            if chunk.completion_tokens:
                engine_completion_tokens = chunk.completion_tokens
            if chunk.rejected_tool_names:
                rejected_tool_names.extend(chunk.rejected_tool_names)
            if chunk.tool_calls:
                tool_names.extend(_names_from_tool_calls(chunk.tool_calls))
                streamed_tool_calls.extend(chunk.tool_calls)
            delta: dict[str, Any] = {}
            if chunk.text:
                delta["content"] = chunk.text
            if chunk.tool_calls:
                delta["tool_calls"] = chunk.tool_calls
            data = json.dumps({
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": delta,
                        "finish_reason": chunk.finish_reason,
                    }
                ],
            })
            yield f"data: {data}\n\n"

        # Final usage chunk (only when requested)
        if include_usage:
            estimated_completion_tokens = max(1, len(total_text) // 4)
            usage_data = json.dumps({
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [],
                "usage": {
                    "prompt_tokens": estimated_prompt_tokens,
                    "completion_tokens": estimated_completion_tokens,
                    "total_tokens": estimated_prompt_tokens + estimated_completion_tokens,
                },
            })
            yield f"data: {usage_data}\n\n"

        yield "data: [DONE]\n\n"

    except (asyncio.CancelledError, GeneratorExit):
        status = "cancelled"
        finish_reason = finish_reason or "cancelled"
    except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
        status = "cancelled"
        finish_reason = finish_reason or "cancelled"
    except Exception as exc:
        status = "error"
        error = str(exc)
        finish_reason = finish_reason or "error"
    finally:
        if engine_completion_tokens is not None:
            logged_completion_tokens = engine_completion_tokens
            token_count_source = "engine"
        else:
            logged_completion_tokens = max(0, len(total_text) // 4)
            token_count_source = "estimated"
        logged_prompt_tokens = engine_prompt_tokens or estimated_prompt_tokens
        duration = time.time() - start_time
        tokens_per_second = logged_completion_tokens / duration if duration > 0 else 0
        await _log_call(
            metrics_service,
            prompt_tokens=logged_prompt_tokens,
            completion_tokens=logged_completion_tokens,
            tokens_per_second=tokens_per_second,
            request_id=request_id,
            model=model,
            endpoint="chat.completions",
            stream=True,
            status=status,
            duration_ms=round(duration * 1000, 1),
            finish_reason=finish_reason,
            has_tools=tools_are_active(request.tools, request.tool_choice),
            tool_choice=_tool_choice_label(request.tool_choice),
            tool_names=tool_names,
            rejected_tool_names=rejected_tool_names,
            **_call_body_fields(
                request,
                status=status,
                finish_reason=finish_reason,
                engine_finish_reason=engine_finish_reason or finish_reason,
                completion_text=total_text,
                prompt_tokens=logged_prompt_tokens,
                completion_tokens=logged_completion_tokens,
                token_count_source=token_count_source,
                tool_calls=streamed_tool_calls or None,
                error=error,
            ),
        )


@router.post("/completions")
async def completions(
    request: Request,
    body: CompletionRequestModel,
    _: str = Depends(verify_api_key),
):
    """Create a completion."""
    ensure_inference_mode(request)
    backend_manager = get_backend_manager(request)
    config_manager = get_config_manager(request)

    selected_model = _resolve_loaded_model(backend_manager, body.model)
    model_id = selected_model.id

    # Get per-model configuration defaults
    model_config = None
    if config_manager:
        model_config = await config_manager.get_model_config(model_id)

    _, config = build_generation_config(
        model_config=model_config,
        sampling_profile="text",
        temperature=body.temperature,
        top_p=body.top_p,
        top_k=body.top_k,
        repetition_penalty=body.repetition_penalty,
        frequency_penalty=body.frequency_penalty,
        presence_penalty=body.presence_penalty,
        max_tokens=body.max_tokens,
        max_completion_tokens=body.max_completion_tokens,
        stop=body.stop,
    )
    completion_request = CompletionRequest(
        prompt=body.prompt,
        config=config,
        stream=body.stream or False,
        enable_thinking=_resolve_enable_thinking(
            body.enable_thinking, body.chat_template_kwargs
        ),
    )

    request_id = f"cmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())
    model_name = model_id

    if body.stream:
        # Pre-check queue availability for streaming requests
        available, queue_len = backend_manager.check_queue_available(model_id)
        if not available:
            raise HTTPException(
                status_code=503,
                detail=f"Request queue is full. {queue_len} requests waiting. Please try again later."
            )

        metrics = get_metrics_service(request)
        include_usage = (
            body.stream_options.include_usage
            if body.stream_options and body.stream_options.include_usage
            else False
        )
        # Estimate prompt tokens from prompt text for streaming
        estimated_prompt_tokens = max(1, len(body.prompt) // 4)
        return StreamingResponse(
            _stream_completion(
                backend_manager, metrics, completion_request, request_id, created, model_name,
                estimated_prompt_tokens, include_usage, model_id=model_id,
            ),
            media_type="text/event-stream",
        )

    metrics = get_metrics_service(request)
    try:
        start_time = time.time()
        response = await backend_manager.generate(completion_request, model_id=model_id)
        duration = time.time() - start_time
    except RuntimeError as e:
        await _log_call(
            metrics,
            request_id=request_id,
            model=model_name,
            endpoint="completions",
            stream=False,
            status="error",
            **_call_body_fields(
                completion_request,
                status="error",
                finish_reason="error",
                completion_text="",
                error=str(e),
            ),
        )
        if "queue" in str(e).lower():
            raise HTTPException(status_code=503, detail=str(e))
        raise HTTPException(status_code=500, detail=str(e))

    tokens_per_second = response.completion_tokens / duration if duration > 0 else 0
    await _log_call(
        metrics,
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        tokens_per_second=tokens_per_second,
        request_id=request_id,
        model=model_name,
        endpoint="completions",
        stream=False,
        status="ok",
        duration_ms=round(duration * 1000, 1),
        finish_reason=response.finish_reason,
        **_call_body_fields(
            completion_request,
            status="ok",
            finish_reason=response.finish_reason,
            engine_finish_reason=response.engine_finish_reason or response.finish_reason,
            completion_text=response.text or "",
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            token_count_source="engine",
        ),
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
    estimated_prompt_tokens: int = 0, include_usage: bool = False, model_id: Optional[str] = None,
):
    """Stream completion chunks per OpenAI spec."""
    import asyncio

    total_text = ""
    start_time = time.time()
    finish_reason = None
    status = "ok"
    error = None
    engine_prompt_tokens: Optional[int] = None
    engine_completion_tokens: Optional[int] = None
    engine_finish_reason: Optional[str] = None

    try:
        async for chunk in backend_manager.generate_stream(request, model_id=model_id):
            if chunk.text:
                total_text += chunk.text
            if chunk.finish_reason:
                finish_reason = chunk.finish_reason
            if chunk.engine_finish_reason:
                engine_finish_reason = chunk.engine_finish_reason
            if chunk.prompt_tokens:
                engine_prompt_tokens = chunk.prompt_tokens
            if chunk.completion_tokens:
                engine_completion_tokens = chunk.completion_tokens
            data = json.dumps({
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
            })
            yield f"data: {data}\n\n"

        # Final usage chunk (only when requested)
        if include_usage:
            estimated_completion_tokens = max(1, len(total_text) // 4)
            usage_data = json.dumps({
                "id": request_id,
                "object": "text_completion",
                "created": created,
                "model": model,
                "choices": [],
                "usage": {
                    "prompt_tokens": estimated_prompt_tokens,
                    "completion_tokens": estimated_completion_tokens,
                    "total_tokens": estimated_prompt_tokens + estimated_completion_tokens,
                },
            })
            yield f"data: {usage_data}\n\n"

        yield "data: [DONE]\n\n"

    except (asyncio.CancelledError, GeneratorExit):
        status = "cancelled"
        finish_reason = finish_reason or "cancelled"
    except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
        status = "cancelled"
        finish_reason = finish_reason or "cancelled"
    except Exception as exc:
        status = "error"
        error = str(exc)
        finish_reason = finish_reason or "error"
    finally:
        if engine_completion_tokens is not None:
            logged_completion_tokens = engine_completion_tokens
            token_count_source = "engine"
        else:
            logged_completion_tokens = max(0, len(total_text) // 4)
            token_count_source = "estimated"
        logged_prompt_tokens = engine_prompt_tokens or estimated_prompt_tokens
        duration = time.time() - start_time
        tokens_per_second = logged_completion_tokens / duration if duration > 0 else 0
        await _log_call(
            metrics_service,
            prompt_tokens=logged_prompt_tokens,
            completion_tokens=logged_completion_tokens,
            tokens_per_second=tokens_per_second,
            request_id=request_id,
            model=model,
            endpoint="completions",
            stream=True,
            status=status,
            duration_ms=round(duration * 1000, 1),
            finish_reason=finish_reason,
            **_call_body_fields(
                request,
                status=status,
                finish_reason=finish_reason,
                engine_finish_reason=engine_finish_reason or finish_reason,
                completion_text=total_text,
                prompt_tokens=logged_prompt_tokens,
                completion_tokens=logged_completion_tokens,
                token_count_source=token_count_source,
                error=error,
            ),
        )
