"""Base backend interface for LLM inference."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional, Union


@dataclass
class ChatMessage:
    """A chat message."""

    role: str  # "system", "user", "assistant", "tool", "function"
    content: Optional[Union[str, list]] = None
    tool_calls: Optional[list] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


def message_to_openai(message: ChatMessage) -> dict:
    """Convert an internal chat message to OpenAI / chat-template shape."""
    payload: dict = {"role": message.role}
    if message.content is not None:
        payload["content"] = message.content
    elif not message.tool_calls:
        payload["content"] = ""
    if message.tool_calls:
        payload["tool_calls"] = message.tool_calls
    if message.tool_call_id:
        payload["tool_call_id"] = message.tool_call_id
    if message.name:
        payload["name"] = message.name
    return payload


@dataclass
class GenerationConfig:
    """Configuration for text generation."""

    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    stop: list[str] = field(default_factory=list)


@dataclass
class CompletionRequest:
    """Request for completion."""

    prompt: Optional[str] = None
    messages: Optional[list[ChatMessage]] = None
    config: GenerationConfig = field(default_factory=GenerationConfig)
    stream: bool = False
    # None = model/template default; True/False forces thinking on or off when supported
    enable_thinking: Optional[bool] = None
    tools: Optional[list] = None
    tool_choice: Optional[object] = None


@dataclass
class CompletionResponse:
    """Response from completion."""

    text: str
    finish_reason: str = "stop"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_content: Optional[str] = None
    tool_calls: Optional[list] = None
    rejected_tool_names: Optional[list[str]] = None
    engine_finish_reason: Optional[str] = None


@dataclass
class StreamChunk:
    """A chunk of streamed response."""

    text: str = ""
    finish_reason: Optional[str] = None
    tool_calls: Optional[list] = None
    rejected_tool_names: Optional[list[str]] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    engine_finish_reason: Optional[str] = None


# Qwen/DeepSeek reasoning delimiters. Later Qwen chat templates inject the
# opener, so generation often contains only the closer.
THINK_TAG_PAIRS: tuple[tuple[str, str], ...] = (
    ("<think>", "</think>"),
    ("<redacted_thinking>", "</redacted_thinking>"),
    ("<thinking>", "</thinking>"),
)
THINK_TAGS: frozenset[str] = frozenset(tag for pair in THINK_TAG_PAIRS for tag in pair)


def split_thinking_content(text: str) -> tuple[Optional[str], str]:
    """Split model output into (reasoning, answer) using common think tags.

    Handles a missing opener (Qwen templates inject ``<think>`` into the
    prompt, so the model may only emit ``</think>``). Returns
    ``(None, text)`` when no thinking block is present.
    """
    if not text:
        return None, text

    best: Optional[tuple[int, int, int, str, str]] = None
    for open_tag, close_tag in THINK_TAG_PAIRS:
        open_idx = text.find(open_tag)
        close_idx = text.find(close_tag)
        if open_idx == -1 and close_idx == -1:
            continue
        if open_idx == -1:
            signal = close_idx
        elif close_idx == -1:
            signal = open_idx
        else:
            signal = min(open_idx, close_idx)
        if best is None or signal < best[0]:
            best = (signal, open_idx, close_idx, open_tag, close_tag)

    if best is None:
        return None, text

    _, open_idx, close_idx, open_tag, close_tag = best

    if open_idx != -1:
        after_open = open_idx + len(open_tag)
        prefix = text[:open_idx]
        if close_idx == -1 or close_idx < after_open:
            reasoning = text[after_open:].strip()
            return (reasoning or None), prefix.strip()
        reasoning = text[after_open:close_idx].strip()
        answer = text[close_idx + len(close_tag) :].lstrip("\n")
        prefix = prefix.strip()
        if prefix:
            answer = f"{prefix}\n{answer}" if answer else prefix
        return (reasoning or None), answer

    reasoning = text[:close_idx].strip()
    answer = text[close_idx + len(close_tag) :].lstrip("\n")
    return (reasoning or None), answer


class BaseBackend(ABC):
    """Abstract base class for LLM backends."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    @abstractmethod
    async def load(self) -> None:
        """Load the model into memory."""
        pass

    @abstractmethod
    async def unload(self) -> None:
        """Unload the model from memory."""
        pass

    @abstractmethod
    async def generate(self, request: CompletionRequest) -> CompletionResponse:
        """Generate a completion."""
        pass

    @abstractmethod
    async def generate_stream(
        self, request: CompletionRequest
    ) -> AsyncIterator[StreamChunk]:
        """Generate a streaming completion."""
        pass

    def format_chat_prompt(self, messages: list[ChatMessage]) -> str:
        """Format chat messages into a prompt string.

        Override this in subclasses for model-specific formatting.
        This is a generic ChatML format.
        """
        formatted = ""
        for msg in messages:
            if msg.role == "system":
                formatted += f"<|im_start|>system\n{msg.content}<|im_end|>\n"
            elif msg.role == "user":
                formatted += f"<|im_start|>user\n{msg.content}<|im_end|>\n"
            elif msg.role == "assistant":
                formatted += f"<|im_start|>assistant\n{msg.content}<|im_end|>\n"
        formatted += "<|im_start|>assistant\n"
        return formatted

    @abstractmethod
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        pass
