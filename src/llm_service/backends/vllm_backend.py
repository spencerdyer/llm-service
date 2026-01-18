"""vLLM backend for Linux/CUDA systems."""

import asyncio
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
    """vLLM-based backend for CUDA systems."""

    def __init__(self, model_path: str, quantization: Optional[str] = None):
        super().__init__(model_path)
        self._engine = None
        self._tokenizer = None
        self._quantization = quantization

    async def load(self) -> None:
        """Load the vLLM model."""
        if self._loaded:
            return

        def _load():
            from vllm import LLM

            # Configure based on quantization
            kwargs = {
                "model": self.model_path,
                "trust_remote_code": True,
                "max_model_len": settings.max_model_len,
                "gpu_memory_utilization": settings.gpu_memory_utilization,
            }

            if self._quantization:
                if self._quantization.lower() == "awq":
                    kwargs["quantization"] = "awq"
                elif self._quantization.lower() == "gptq":
                    kwargs["quantization"] = "gptq"

            engine = LLM(**kwargs)
            return engine

        self._engine = await asyncio.to_thread(_load)
        self._tokenizer = self._engine.get_tokenizer()
        self._loaded = True

    async def unload(self) -> None:
        """Unload the model."""
        # vLLM doesn't have a clean unload mechanism
        # Setting to None and relying on GC
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

    async def generate(self, request: CompletionRequest) -> CompletionResponse:
        """Generate a completion."""
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        # Build prompt
        if request.messages:
            prompt = self._apply_chat_template(request.messages)
        elif request.prompt:
            prompt = request.prompt
        else:
            raise ValueError("Either prompt or messages must be provided")

        def _generate():
            from vllm import SamplingParams

            sampling_params = SamplingParams(
                max_tokens=request.config.max_tokens,
                temperature=request.config.temperature,
                top_p=request.config.top_p,
                top_k=request.config.top_k,
                repetition_penalty=request.config.repetition_penalty,
                stop=request.config.stop if request.config.stop else None,
            )

            outputs = self._engine.generate([prompt], sampling_params)
            return outputs[0]

        output = await asyncio.to_thread(_generate)

        return CompletionResponse(
            text=output.outputs[0].text,
            finish_reason=output.outputs[0].finish_reason or "stop",
            prompt_tokens=len(output.prompt_token_ids),
            completion_tokens=len(output.outputs[0].token_ids),
            total_tokens=len(output.prompt_token_ids) + len(output.outputs[0].token_ids),
        )

    async def generate_stream(
        self, request: CompletionRequest
    ) -> AsyncIterator[StreamChunk]:
        """Generate a streaming completion using vLLM's async engine."""
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        # Build prompt
        if request.messages:
            prompt = self._apply_chat_template(request.messages)
        elif request.prompt:
            prompt = request.prompt
        else:
            raise ValueError("Either prompt or messages must be provided")

        # For streaming with vLLM, we need to use the AsyncLLMEngine
        # This is a simplified implementation using the sync engine with chunking
        import queue
        import threading

        token_queue: queue.Queue[Optional[tuple[str, Optional[str]]]] = queue.Queue()

        def _stream_generate():
            from vllm import SamplingParams

            sampling_params = SamplingParams(
                max_tokens=request.config.max_tokens,
                temperature=request.config.temperature,
                top_p=request.config.top_p,
                top_k=request.config.top_k,
                repetition_penalty=request.config.repetition_penalty,
                stop=request.config.stop if request.config.stop else None,
            )

            try:
                # vLLM sync generate doesn't stream, so we generate all at once
                # and then yield tokens. For true streaming, use AsyncLLMEngine.
                outputs = self._engine.generate([prompt], sampling_params)
                output = outputs[0]
                text = output.outputs[0].text
                finish_reason = output.outputs[0].finish_reason or "stop"

                # Yield in chunks to simulate streaming
                chunk_size = 4
                for i in range(0, len(text), chunk_size):
                    chunk = text[i : i + chunk_size]
                    is_last = i + chunk_size >= len(text)
                    token_queue.put((chunk, finish_reason if is_last else None))

                if len(text) == 0:
                    token_queue.put(("", finish_reason))
            except Exception as e:
                token_queue.put(("", f"error: {e}"))
            finally:
                token_queue.put(None)

        # Start generation in background thread
        thread = threading.Thread(target=_stream_generate)
        thread.start()

        # Yield tokens as they arrive
        while True:
            try:
                result = await asyncio.to_thread(token_queue.get, timeout=60)
                if result is None:
                    break
                text, finish_reason = result
                yield StreamChunk(text=text, finish_reason=finish_reason)
                if finish_reason:
                    break
            except queue.Empty:
                yield StreamChunk(text="", finish_reason="timeout")
                break

        thread.join(timeout=1)

    def get_model_info(self) -> dict:
        """Get model information."""
        info = {
            "backend": "vllm",
            "model_path": self.model_path,
            "loaded": self._loaded,
            "quantization": self._quantization,
        }

        if self._engine is not None:
            try:
                model_config = self._engine.llm_engine.model_config
                info["max_model_len"] = model_config.max_model_len
                info["dtype"] = str(model_config.dtype)
            except Exception:
                pass

        return info
