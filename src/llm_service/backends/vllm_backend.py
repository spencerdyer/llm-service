"""vLLM backend for Linux/CUDA systems."""

import asyncio
import uuid
from typing import AsyncIterator, Optional

from llm_service.backends.base import (
    THINK_TAGS,
    BaseBackend,
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    StreamChunk,
    message_to_openai,
    split_thinking_content,
)
from llm_service.backends.multimodal import collect_multimodal_data
from llm_service.backends.tool_parse import (
    apply_tool_policy,
    guided_tool_schema,
    parse_tool_calls_from_text,
    tool_calls_to_stream_deltas,
    tools_are_active,
)
from llm_service.config import settings


# Map our stored quantization labels to vLLM AsyncEngineArgs.quantization values.
# NVFP4/modelopt checkpoints usually self-describe via hf_quant_config.json; we still
# pass an explicit method when vLLM expects one.
_VLLM_QUANTIZATION_MAP = {
    "awq": "awq",
    "gptq": "gptq",
    "nvfp4": "modelopt",
    "modelopt": "modelopt",
}


class VLLMBackend(BaseBackend):
    """vLLM-based backend for CUDA systems with true streaming and concurrent request support."""

    def __init__(
        self,
        model_path: str,
        quantization: Optional[str] = None,
        max_model_len: Optional[int] = None,
    ):
        super().__init__(model_path)
        self._engine = None
        self._tokenizer = None
        self._quantization = quantization
        self._max_model_len = max_model_len or settings.max_model_len
        self._active_requests: set[str] = set()

    def _vllm_quantization_arg(self) -> Optional[str]:
        """Return a vLLM quantization method string, or None to let vLLM auto-detect."""
        if not self._quantization:
            return None
        return _VLLM_QUANTIZATION_MAP.get(self._quantization.lower())

    async def load(self) -> None:
        """Load the vLLM model using AsyncLLMEngine for streaming support."""
        if self._loaded:
            return

        from vllm import AsyncEngineArgs, AsyncLLMEngine

        engine_kwargs = {
            "model": self.model_path,
            "trust_remote_code": True,
            "max_model_len": self._max_model_len,
            "gpu_memory_utilization": settings.gpu_memory_utilization,
            "disable_log_stats": True,
            "limit_mm_per_prompt": {
                "image": {"count": 4, "width": 1024, "height": 1024},
                "video": {"count": 1, "num_frames": 8, "width": 640, "height": 360},
            },
            "allowed_local_media_path": "/",
        }
        quantization = self._vllm_quantization_arg()
        if quantization:
            engine_kwargs["quantization"] = quantization
        tool_kwargs = self._tool_engine_kwargs()
        engine_kwargs.update(tool_kwargs)

        try:
            engine_args = AsyncEngineArgs(**engine_kwargs)
        except TypeError:
            for key in tool_kwargs:
                engine_kwargs.pop(key, None)
            engine_args = AsyncEngineArgs(**engine_kwargs)

        # Create async engine - must be done in async context
        # Use from_engine_args which handles initialization properly
        self._engine = AsyncLLMEngine.from_engine_args(engine_args)

        # Get tokenizer - access through the engine's internal structures
        try:
            # Try V1 engine path first
            if hasattr(self._engine, 'engine') and self._engine.engine is not None:
                if hasattr(self._engine.engine, 'tokenizer'):
                    tok = self._engine.engine.tokenizer
                    self._tokenizer = tok.tokenizer if hasattr(tok, 'tokenizer') else tok
                elif hasattr(self._engine.engine, 'get_tokenizer'):
                    self._tokenizer = self._engine.engine.get_tokenizer()
        except Exception:
            pass

        # Fallback: load tokenizer separately
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path, trust_remote_code=True
                )
            except Exception:
                pass

        self._loaded = True

    async def unload(self) -> None:
        """Unload the model."""
        if self._engine is not None:
            # Cancel any active requests
            for req_id in list(self._active_requests):
                try:
                    await self._engine.abort(req_id)
                except Exception:
                    pass
            self._active_requests.clear()

            # Shutdown the async engine properly
            try:
                if hasattr(self._engine, 'shutdown_background_loop'):
                    self._engine.shutdown_background_loop()
            except Exception:
                pass

        self._engine = None
        self._tokenizer = None
        self._loaded = False

    def _apply_chat_template(
        self,
        messages: list[ChatMessage],
        enable_thinking: Optional[bool] = None,
        tools: Optional[list] = None,
    ) -> str:
        """Apply the tokenizer's chat template if available."""
        if self._tokenizer is not None and hasattr(self._tokenizer, "apply_chat_template"):
            message_dicts = [message_to_openai(m) for m in messages]
            kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if enable_thinking is not None:
                kwargs["enable_thinking"] = enable_thinking
            if tools:
                kwargs["tools"] = tools
            try:
                return self._tokenizer.apply_chat_template(message_dicts, **kwargs)
            except TypeError:
                # Older tokenizers may only accept template vars via chat_template_kwargs
                if enable_thinking is not None:
                    try:
                        return self._tokenizer.apply_chat_template(
                            message_dicts,
                            tokenize=False,
                            add_generation_prompt=True,
                            chat_template_kwargs={"enable_thinking": enable_thinking},
                        )
                    except TypeError:
                        pass
                try:
                    return self._tokenizer.apply_chat_template(
                        message_dicts, tokenize=False, add_generation_prompt=True
                    )
                except Exception:
                    pass
            except Exception:
                pass
        return self.format_chat_prompt(messages)

    def _build_prompt(self, request: CompletionRequest) -> str:
        """Build prompt from request."""
        if request.messages:
            return self._apply_chat_template(
                request.messages,
                enable_thinking=request.enable_thinking,
                tools=request.tools,
            )
        elif request.prompt:
            return request.prompt
        else:
            raise ValueError("Either prompt or messages must be provided")

    def _engine_input(self, request: CompletionRequest):
        """Build a vLLM prompt, attaching images/videos when present."""
        prompt = self._build_prompt(request)
        mm_data = collect_multimodal_data(request.messages)
        if not mm_data:
            return prompt
        return {"prompt": prompt, "multi_modal_data": mm_data}

    @staticmethod
    def _tool_engine_kwargs() -> dict:
        """Enable vLLM native tool calling when this version supports it."""
        extra: dict = {}
        try:
            from vllm import AsyncEngineArgs
            import inspect

            params = inspect.signature(AsyncEngineArgs.__init__).parameters
        except Exception:
            return extra
        if "enable_auto_tool_choice" in params:
            extra["enable_auto_tool_choice"] = True
        if "tool_call_parser" in params:
            extra["tool_call_parser"] = "hermes"
        return extra

    def _guided_decoding_kwargs(self, request: CompletionRequest) -> dict:
        """Constrain generation to an allowed tool JSON object when required."""
        if request.enable_thinking:
            return {}
        schema = guided_tool_schema(request.tools, request.tool_choice)
        if not schema:
            return {}
        try:
            from vllm.sampling_params import GuidedDecodingParams

            return {"guided_decoding": GuidedDecodingParams(json=schema)}
        except Exception:
            pass
        try:
            from vllm.sampling_params import StructuredOutputsParams

            return {"structured_outputs": StructuredOutputsParams(json=schema)}
        except Exception:
            return {}

    def _create_sampling_params(self, request: CompletionRequest):
        """Create vLLM SamplingParams from request config."""
        from vllm import SamplingParams

        kwargs = {
            "max_tokens": request.config.max_tokens,
            "temperature": request.config.temperature,
            "top_p": request.config.top_p,
            "top_k": request.config.top_k if request.config.top_k > 0 else -1,
            "repetition_penalty": request.config.repetition_penalty,
            "stop": request.config.stop if request.config.stop else None,
        }
        kwargs.update(self._guided_decoding_kwargs(request))
        try:
            return SamplingParams(**kwargs)
        except TypeError:
            kwargs.pop("guided_decoding", None)
            kwargs.pop("structured_outputs", None)
            return SamplingParams(**kwargs)

    def _finalize_tools(self, text: str, request: CompletionRequest):
        """Parse and sanitize tool calls against the request allowlist."""
        if not tools_are_active(request.tools, request.tool_choice):
            leftover, _ = parse_tool_calls_from_text(text)
            return leftover or text, None, "stop", []

        leftover, parsed = parse_tool_calls_from_text(text)
        return apply_tool_policy(
            leftover if parsed else text,
            parsed,
            request.tools,
            request.tool_choice,
        )

    def _decode_output_text(self, completion) -> str:
        """Decode completion text while keeping think delimiters.

        vLLM skips special tokens by default; Qwen 3.x marks ``<think>`` /
        ``</think>`` as special, which concatenates reasoning with the answer.
        """
        token_ids = getattr(completion, "token_ids", None)
        if self._tokenizer is None or not token_ids:
            return completion.text or ""
        try:
            raw = self._tokenizer.decode(list(token_ids), skip_special_tokens=False)
        except Exception:
            return completion.text or ""
        for tok in getattr(self._tokenizer, "all_special_tokens", []) or []:
            if tok and tok not in THINK_TAGS:
                raw = raw.replace(tok, "")
        return raw

    @property
    def queue_length(self) -> int:
        """Get the number of active requests."""
        return len(self._active_requests)

    @property
    def is_queue_full(self) -> bool:
        """Check if we've hit a reasonable concurrent request limit."""
        # vLLM handles batching internally, but we can set a soft limit
        return len(self._active_requests) >= 64

    async def generate(self, request: CompletionRequest) -> CompletionResponse:
        """Generate a completion."""
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        prompt = self._engine_input(request)
        sampling_params = self._create_sampling_params(request)
        request_id = str(uuid.uuid4())

        self._active_requests.add(request_id)
        try:
            # Collect all output from streaming
            full_text = ""
            finish_reason = "stop"
            prompt_tokens = 0
            completion_tokens = 0

            async for output in self._engine.generate(prompt, sampling_params, request_id):
                if output.outputs:
                    full_text = self._decode_output_text(output.outputs[0])
                    finish_reason = output.outputs[0].finish_reason or "stop"
                    completion_tokens = len(output.outputs[0].token_ids)
                if hasattr(output, 'prompt_token_ids'):
                    prompt_tokens = len(output.prompt_token_ids)

            leftover, tool_calls, tool_finish, rejected = self._finalize_tools(
                full_text, request
            )
            engine_finish = finish_reason
            if tool_calls:
                full_text = leftover
                finish_reason = tool_finish
            elif rejected:
                full_text = leftover
                finish_reason = "stop"

            return CompletionResponse(
                text=full_text,
                finish_reason=finish_reason,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                tool_calls=tool_calls,
                rejected_tool_names=rejected or None,
                engine_finish_reason=engine_finish,
            )
        finally:
            self._active_requests.discard(request_id)

    async def generate_stream(
        self, request: CompletionRequest
    ) -> AsyncIterator[StreamChunk]:
        """Generate a streaming completion with true token-by-token streaming."""
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        prompt = self._engine_input(request)
        sampling_params = self._create_sampling_params(request)
        request_id = str(uuid.uuid4())

        self._active_requests.add(request_id)
        previous_text = ""
        buffer_for_tools = tools_are_active(request.tools, request.tool_choice)
        collected_text = ""
        finish_reason = "stop"
        prompt_tokens = 0
        completion_tokens = 0

        try:
            async for output in self._engine.generate(prompt, sampling_params, request_id):
                if hasattr(output, "prompt_token_ids") and output.prompt_token_ids:
                    prompt_tokens = len(output.prompt_token_ids)
                if output.outputs:
                    current_text = self._decode_output_text(output.outputs[0])
                    raw_finish = output.outputs[0].finish_reason
                    if raw_finish:
                        finish_reason = raw_finish
                    completion_tokens = len(output.outputs[0].token_ids)

                    usage = {}
                    if raw_finish:
                        usage = {
                            "prompt_tokens": prompt_tokens or None,
                            "completion_tokens": completion_tokens or None,
                            "engine_finish_reason": raw_finish,
                        }

                    if buffer_for_tools:
                        collected_text = current_text
                        continue

                    if len(current_text) > len(previous_text):
                        delta = current_text[len(previous_text):]
                        previous_text = current_text
                        yield StreamChunk(
                            text=delta,
                            finish_reason=raw_finish,
                            **usage,
                        )
                    elif raw_finish:
                        yield StreamChunk(text="", finish_reason=raw_finish, **usage)

            if buffer_for_tools:
                leftover, parsed, tool_finish, rejected = self._finalize_tools(
                    collected_text, request
                )
                usage = {
                    "prompt_tokens": prompt_tokens or None,
                    "completion_tokens": completion_tokens or None,
                    "engine_finish_reason": finish_reason,
                }
                if parsed:
                    _, answer = split_thinking_content(leftover)
                    if answer:
                        yield StreamChunk(text=answer)
                    for delta in tool_calls_to_stream_deltas(parsed):
                        yield StreamChunk(tool_calls=delta)
                    yield StreamChunk(
                        finish_reason=tool_finish,
                        rejected_tool_names=rejected or None,
                        **usage,
                    )
                else:
                    emit_text = leftover if rejected else collected_text
                    if emit_text:
                        yield StreamChunk(text=emit_text)
                    yield StreamChunk(
                        finish_reason=finish_reason if not rejected else "stop",
                        rejected_tool_names=rejected or None,
                        **usage,
                    )

        except asyncio.CancelledError:
            # Request was cancelled, abort it in vLLM
            try:
                await self._engine.abort(request_id)
            except Exception:
                pass
            raise
        except Exception as e:
            yield StreamChunk(text="", finish_reason=f"error: {e}")
        finally:
            self._active_requests.discard(request_id)

    async def abort_request(self, request_id: str) -> None:
        """Abort an in-progress request."""
        if self._engine is not None and request_id in self._active_requests:
            try:
                await self._engine.abort(request_id)
            except Exception:
                pass
            self._active_requests.discard(request_id)

    def get_model_info(self) -> dict:
        """Get model information."""
        info = {
            "backend": "vllm",
            "model_path": self.model_path,
            "loaded": self._loaded,
            "quantization": self._quantization,
            "max_model_len": self._max_model_len,
            "streaming": True,
            "batching_enabled": True,
            "active_requests": len(self._active_requests),
        }

        if self._engine is not None:
            try:
                # Try to get model config from engine
                if hasattr(self._engine, 'engine') and self._engine.engine is not None:
                    model_config = self._engine.engine.model_config
                    info["max_model_len"] = model_config.max_model_len
                    info["dtype"] = str(model_config.dtype)
            except Exception:
                pass

        return info
