"""GGUF backend for running quantized GGUF models via llama-cpp-python."""

import asyncio
import logging
from pathlib import Path
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
from llm_service.backends.multimodal import text_from_content
from llm_service.backends.tool_parse import (
    apply_tool_policy,
    parse_tool_calls_from_text,
    tool_calls_to_stream_deltas,
    tools_are_active,
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


def _is_think_special_bytes(piece: bytes) -> bool:
    """Return True if a special-token decode is a reasoning delimiter."""
    try:
        text = piece.decode("utf-8").strip()
    except UnicodeDecodeError:
        return False
    return text in THINK_TAGS


def patch_llama_detokenize_keep_think_tags(llm) -> None:
    """Keep ``<think>`` / ``</think>`` when llama.cpp skips special tokens.

    Qwen 3.x marks those tags as special tokens. llama-cpp-python detokenizes
    with ``special=False``, which strips them and concatenates reasoning with
    the answer so the UI cannot fold it.
    """
    tokenizer = getattr(llm, "tokenizer_", None)
    if tokenizer is None or getattr(tokenizer, "_keep_think_patched", False):
        return

    original = tokenizer.detokenize

    def detokenize(tokens, prev_tokens=None, special=False):
        if special:
            return original(tokens, prev_tokens=prev_tokens, special=True)
        if not tokens:
            return original(tokens, prev_tokens=prev_tokens, special=False)

        with_special = original(tokens, prev_tokens=prev_tokens, special=True)
        without = original(tokens, prev_tokens=prev_tokens, special=False)
        if with_special == without:
            return without

        pieces: list[bytes] = []
        prefix = list(prev_tokens) if prev_tokens is not None else []
        for token in tokens:
            kept = original([token], prev_tokens=prefix, special=True)
            dropped = original([token], prev_tokens=prefix, special=False)
            prefix.append(token)
            if dropped:
                pieces.append(dropped)
            elif _is_think_special_bytes(kept):
                pieces.append(kept)
        return b"".join(pieces)

    tokenizer.detokenize = detokenize
    tokenizer._keep_think_patched = True


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
        patch_llama_detokenize_keep_think_tags(self._llm)
        self._loaded = True

    async def unload(self) -> None:
        if self._llm is not None:
            del self._llm
            self._llm = None
        self._loaded = False

    def _build_messages(self, request: CompletionRequest) -> list[dict]:
        if request.messages:
            messages = [message_to_openai(m) for m in request.messages]
            for message in messages:
                content = message.get("content")
                if isinstance(content, list):
                    message["content"] = text_from_content(content)
            return messages
        if request.prompt:
            return [{"role": "user", "content": request.prompt}]
        raise ValueError("Either prompt or messages must be provided")

    def _chat_kwargs(self, request: CompletionRequest, *, stream: bool) -> dict:
        config = request.config
        kwargs = {
            "messages": self._build_messages(request),
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "repeat_penalty": config.repetition_penalty,
            "stop": config.stop or None,
            "stream": stream,
        }
        if request.tools:
            kwargs["tools"] = request.tools
        if request.tool_choice is not None:
            kwargs["tool_choice"] = request.tool_choice
        return kwargs

    async def generate(self, request: CompletionRequest) -> CompletionResponse:
        if not self._loaded or self._llm is None:
            raise RuntimeError("Model not loaded")

        kwargs = self._chat_kwargs(request, stream=False)

        def _generate():
            return self._llm.create_chat_completion(**kwargs)

        result = await asyncio.to_thread(_generate)

        text = ""
        finish_reason = "stop"
        tool_calls = None
        if result.get("choices"):
            choice = result["choices"][0]
            message = choice.get("message") or {}
            text = message.get("content") or ""
            finish_reason = choice.get("finish_reason", "stop") or "stop"
            tool_calls = message.get("tool_calls") or None
            if not tool_calls:
                leftover, parsed = parse_tool_calls_from_text(text)
                if parsed:
                    text = leftover
                    tool_calls = parsed
                    finish_reason = "tool_calls"

        leftover, kept, tool_finish, rejected = apply_tool_policy(
            text, tool_calls, request.tools, request.tool_choice
        )
        if kept:
            text, tool_calls, finish_reason = leftover, kept, tool_finish
        else:
            text, tool_calls = leftover, None
            if rejected and finish_reason == "tool_calls":
                finish_reason = "stop"

        usage = result.get("usage", {})
        return CompletionResponse(
            text=text,
            finish_reason=finish_reason,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            tool_calls=tool_calls,
            rejected_tool_names=rejected or None,
        )

    async def generate_stream(
        self, request: CompletionRequest
    ) -> AsyncIterator[StreamChunk]:
        if not self._loaded or self._llm is None:
            raise RuntimeError("Model not loaded")

        kwargs = self._chat_kwargs(request, stream=True)

        import queue
        import threading

        _STREAM_ERROR = object()
        token_queue: queue.Queue = queue.Queue()
        thread_error: list[BaseException] = []

        def _stream():
            try:
                for chunk in self._llm.create_chat_completion(**kwargs):
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

        buffer_for_tools = tools_are_active(request.tools, request.tool_choice)
        collected_text = ""
        native_tool_calls: list = []
        finish_reason = "stop"

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
                        if not buffer_for_tools:
                            yield StreamChunk(text="", finish_reason=finish_reason)
                        break

                    choices = chunk.get("choices", [])
                    if not choices:
                        continue

                    delta = choices[0].get("delta") or {}
                    text = delta.get("content") or ""
                    tool_calls = delta.get("tool_calls") or None
                    finish = choices[0].get("finish_reason")

                    if buffer_for_tools:
                        if text:
                            collected_text += text
                        if tool_calls:
                            native_tool_calls.extend(tool_calls)
                        if finish:
                            finish_reason = finish
                            break
                        continue

                    if text or tool_calls:
                        yield StreamChunk(text=text, tool_calls=tool_calls)
                    if finish:
                        yield StreamChunk(text="", finish_reason=finish)
                        break
                except queue.Empty:
                    finish_reason = "timeout"
                    if not buffer_for_tools:
                        yield StreamChunk(text="", finish_reason="timeout")
                    break

            if buffer_for_tools:
                parsed = native_tool_calls or None
                leftover = collected_text
                if not parsed:
                    leftover, parsed = parse_tool_calls_from_text(collected_text)
                leftover, parsed, tool_finish, rejected = apply_tool_policy(
                    leftover if parsed else collected_text,
                    parsed,
                    request.tools,
                    request.tool_choice,
                )
                if parsed:
                    _, answer = split_thinking_content(leftover)
                    if answer:
                        yield StreamChunk(text=answer)
                    for delta in tool_calls_to_stream_deltas(parsed):
                        yield StreamChunk(tool_calls=delta)
                    yield StreamChunk(
                        finish_reason=tool_finish,
                        rejected_tool_names=rejected or None,
                    )
                else:
                    emit_text = leftover if rejected else collected_text
                    if emit_text:
                        yield StreamChunk(text=emit_text)
                    yield StreamChunk(
                        finish_reason="stop" if rejected else finish_reason,
                        rejected_tool_names=rejected or None,
                    )
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
