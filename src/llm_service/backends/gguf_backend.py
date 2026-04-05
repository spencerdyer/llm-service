"""GGUF backend for running quantized GGUF models via llama-cpp-python."""

import asyncio
import logging
from pathlib import Path
from typing import AsyncIterator, Optional

from llm_service.backends.base import (
    BaseBackend,
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    StreamChunk,
)

logger = logging.getLogger(__name__)


def find_gguf_file(model_path: str) -> str:
    """Resolve the actual .gguf file from a model path.

    The path may point directly to a .gguf file or to a directory
    containing one or more .gguf files.
    """
    p = Path(model_path)
    if p.is_file() and p.suffix == ".gguf":
        return str(p)

    if p.is_dir():
        gguf_files = sorted(p.glob("*.gguf"))
        if not gguf_files:
            raise FileNotFoundError(f"No .gguf files found in {model_path}")
        if len(gguf_files) == 1:
            return str(gguf_files[0])
        # Prefer Q4_K_M > Q5_K_M > Q8_0 > first file
        for preference in ["Q4_K_M", "Q5_K_M", "Q8_0", "Q4_0"]:
            for f in gguf_files:
                if preference in f.name:
                    return str(f)
        return str(gguf_files[0])

    raise FileNotFoundError(f"GGUF model path does not exist: {model_path}")


class GGUFBackend(BaseBackend):
    """llama-cpp-python backend for GGUF models with Metal acceleration."""

    def __init__(self, model_path: str, n_ctx: int = 4096, n_gpu_layers: int = -1):
        super().__init__(model_path)
        self._llm = None
        self._n_ctx = n_ctx
        self._n_gpu_layers = n_gpu_layers
        self._gguf_file: Optional[str] = None

    async def load(self) -> None:
        if self._loaded:
            return

        self._gguf_file = find_gguf_file(self.model_path)

        def _load():
            from llama_cpp import Llama

            return Llama(
                model_path=self._gguf_file,
                n_ctx=self._n_ctx,
                n_gpu_layers=self._n_gpu_layers,
                verbose=False,
            )

        self._llm = await asyncio.to_thread(_load)
        self._loaded = True

    async def unload(self) -> None:
        if self._llm is not None:
            del self._llm
            self._llm = None
        self._loaded = False

    def _build_messages(self, request: CompletionRequest) -> list[dict]:
        if request.messages:
            return [{"role": m.role, "content": m.content} for m in request.messages]
        if request.prompt:
            return [{"role": "user", "content": request.prompt}]
        raise ValueError("Either prompt or messages must be provided")

    async def generate(self, request: CompletionRequest) -> CompletionResponse:
        if not self._loaded or self._llm is None:
            raise RuntimeError("Model not loaded")

        messages = self._build_messages(request)
        config = request.config

        def _generate():
            return self._llm.create_chat_completion(
                messages=messages,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                repeat_penalty=config.repetition_penalty,
                stop=config.stop or None,
                stream=False,
            )

        result = await asyncio.to_thread(_generate)

        text = ""
        finish_reason = "stop"
        if result.get("choices"):
            choice = result["choices"][0]
            text = choice.get("message", {}).get("content", "")
            finish_reason = choice.get("finish_reason", "stop") or "stop"

        usage = result.get("usage", {})
        return CompletionResponse(
            text=text,
            finish_reason=finish_reason,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        )

    async def generate_stream(
        self, request: CompletionRequest
    ) -> AsyncIterator[StreamChunk]:
        if not self._loaded or self._llm is None:
            raise RuntimeError("Model not loaded")

        messages = self._build_messages(request)
        config = request.config

        import queue
        import threading

        _STREAM_ERROR = object()
        token_queue: queue.Queue = queue.Queue()
        thread_error: list[BaseException] = []

        def _stream():
            try:
                for chunk in self._llm.create_chat_completion(
                    messages=messages,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    repeat_penalty=config.repetition_penalty,
                    stop=config.stop or None,
                    stream=True,
                ):
                    token_queue.put(chunk)
            except Exception as exc:
                logger.error("GGUF streaming generation failed: %s", exc, exc_info=True)
                thread_error.append(exc)
                token_queue.put(_STREAM_ERROR)
                return
            finally:
                token_queue.put(None)

        thread = threading.Thread(target=_stream, daemon=True)
        thread.start()

        try:
            while True:
                try:
                    chunk = await asyncio.to_thread(token_queue.get, timeout=60)
                    if chunk is _STREAM_ERROR:
                        err = thread_error[0] if thread_error else RuntimeError("Unknown")
                        raise RuntimeError(
                            f"GGUF generation failed mid-stream: {err}"
                        ) from err
                    if chunk is None:
                        yield StreamChunk(text="", finish_reason="stop")
                        break

                    choices = chunk.get("choices", [])
                    if not choices:
                        continue

                    delta = choices[0].get("delta", {})
                    text = delta.get("content", "")
                    finish = choices[0].get("finish_reason")

                    if text:
                        yield StreamChunk(text=text)
                    if finish:
                        yield StreamChunk(text="", finish_reason=finish)
                        break
                except queue.Empty:
                    yield StreamChunk(text="", finish_reason="timeout")
                    break
        finally:
            thread.join(timeout=2)

    def get_model_info(self) -> dict:
        info = {
            "backend": "gguf",
            "model_path": self.model_path,
            "gguf_file": self._gguf_file,
            "loaded": self._loaded,
            "n_ctx": self._n_ctx,
            "n_gpu_layers": self._n_gpu_layers,
            "batching_enabled": False,
        }
        if self._llm is not None:
            metadata = self._llm.metadata or {}
            info["model_name"] = metadata.get("general.name", "")
            info["model_architecture"] = metadata.get("general.architecture", "")
        return info
