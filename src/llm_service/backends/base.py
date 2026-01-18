"""Base backend interface for LLM inference."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional


@dataclass
class ChatMessage:
    """A chat message."""

    role: str  # "system", "user", "assistant"
    content: str


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


@dataclass
class CompletionResponse:
    """Response from completion."""

    text: str
    finish_reason: str = "stop"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class StreamChunk:
    """A chunk of streamed response."""

    text: str
    finish_reason: Optional[str] = None


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
