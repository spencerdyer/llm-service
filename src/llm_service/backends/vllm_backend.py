"""vLLM backend for Linux/CUDA systems."""

import asyncio
import uuid
from typing import AsyncIterator, Optional

from llm_service.backends.base import (
    BaseBackend,
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    StreamChunk,
)
from llm_service.config import settings


class VLLMBackend(BaseBackend):
    """vLLM-based backend for CUDA systems with true streaming and concurrent request support."""

    def __init__(self, model_path: str, quantization: Optional[str] = None):
        super().__init__(model_path)
        self._engine = None
        self._tokenizer = None
        self._quantization = quantization
        self._active_requests: set[str] = set()

    async def load(self) -> None:
        """Load the vLLM model using AsyncLLMEngine for streaming support."""
        if self._loaded:
            return

        from vllm import AsyncEngineArgs, AsyncLLMEngine

        # Configure engine args
        engine_args = AsyncEngineArgs(
            model=self.model_path,
            trust_remote_code=True,
            max_model_len=settings.max_model_len,
            gpu_memory_utilization=settings.gpu_memory_utilization,
            disable_log_stats=True,
        )

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

    def _apply_chat_template(self, messages: list[ChatMessage]) -> str:
        """Apply the tokenizer's chat template if available."""
        if self._tokenizer is not None and hasattr(self._tokenizer, "apply_chat_template"):
            message_dicts = [{"role": m.role, "content": m.content} for m in messages]
            try:
                return self._tokenizer.apply_chat_template(
                    message_dicts, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass
        return self.format_chat_prompt(messages)

    def _build_prompt(self, request: CompletionRequest) -> str:
        """Build prompt from request."""
        if request.messages:
            return self._apply_chat_template(request.messages)
        elif request.prompt:
            return request.prompt
        else:
            raise ValueError("Either prompt or messages must be provided")

    def _create_sampling_params(self, request: CompletionRequest):
        """Create vLLM SamplingParams from request config."""
        from vllm import SamplingParams

        return SamplingParams(
            max_tokens=request.config.max_tokens,
            temperature=request.config.temperature,
            top_p=request.config.top_p,
            top_k=request.config.top_k if request.config.top_k > 0 else -1,
            repetition_penalty=request.config.repetition_penalty,
            stop=request.config.stop if request.config.stop else None,
        )

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

        prompt = self._build_prompt(request)
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
                    full_text = output.outputs[0].text
                    finish_reason = output.outputs[0].finish_reason or "stop"
                    completion_tokens = len(output.outputs[0].token_ids)
                if hasattr(output, 'prompt_token_ids'):
                    prompt_tokens = len(output.prompt_token_ids)

            return CompletionResponse(
                text=full_text,
                finish_reason=finish_reason,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )
        finally:
            self._active_requests.discard(request_id)

    async def generate_stream(
        self, request: CompletionRequest
    ) -> AsyncIterator[StreamChunk]:
        """Generate a streaming completion with true token-by-token streaming."""
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        prompt = self._build_prompt(request)
        sampling_params = self._create_sampling_params(request)
        request_id = str(uuid.uuid4())

        self._active_requests.add(request_id)
        previous_text = ""

        try:
            async for output in self._engine.generate(prompt, sampling_params, request_id):
                if output.outputs:
                    current_text = output.outputs[0].text
                    finish_reason = output.outputs[0].finish_reason

                    # Yield only the new text (delta)
                    if len(current_text) > len(previous_text):
                        delta = current_text[len(previous_text):]
                        previous_text = current_text
                        yield StreamChunk(
                            text=delta,
                            finish_reason=finish_reason if finish_reason else None
                        )
                    elif finish_reason:
                        # Final chunk with finish reason
                        yield StreamChunk(text="", finish_reason=finish_reason)

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
