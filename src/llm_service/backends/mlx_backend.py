"""MLX backend for Apple Silicon Macs with continuous batching support."""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional, Any

from llm_service.backends.base import (
    BaseBackend,
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    StreamChunk,
)


# Batching configuration
MAX_BATCH_SIZE = 8  # Maximum requests to batch together
BATCH_TIMEOUT_MS = 50  # Max time to wait for more requests to batch (milliseconds)
MAX_QUEUE_SIZE = 100  # Maximum pending requests before rejecting


@dataclass
class PendingRequest:
    """A request waiting to be processed."""
    id: str
    prompt_tokens: list[int]
    prompt_text: str
    config: Any
    future: asyncio.Future
    created_at: float = field(default_factory=time.time)


class MLXBackend(BaseBackend):
    """MLX-based backend for Apple Silicon with continuous batching."""

    def __init__(self, model_path: str):
        super().__init__(model_path)
        self._model = None
        self._tokenizer = None

        # Batching infrastructure
        self._request_queue: asyncio.Queue[PendingRequest] = None
        self._batch_processor_task: Optional[asyncio.Task] = None
        self._shutdown_event: Optional[asyncio.Event] = None
        self._is_processing = False

        # GPU access lock - ensures only one operation uses the GPU at a time
        # This prevents crashes from concurrent Metal command buffer access
        self._gpu_lock: Optional[asyncio.Lock] = None

        # Prompt cache for efficiency
        self._prompt_caches: dict[str, Any] = {}

    async def load(self) -> None:
        """Load the MLX model and start batch processor."""
        if self._loaded:
            return

        def _load():
            from mlx_lm import load
            model, tokenizer = load(self.model_path)
            return model, tokenizer

        self._model, self._tokenizer = await asyncio.to_thread(_load)
        self._loaded = True

        # Initialize batching infrastructure
        self._request_queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
        self._shutdown_event = asyncio.Event()
        self._gpu_lock = asyncio.Lock()
        self._batch_processor_task = asyncio.create_task(self._batch_processor())

    async def unload(self) -> None:
        """Unload the model and stop batch processor."""
        # Stop batch processor
        if self._shutdown_event:
            self._shutdown_event.set()

        if self._batch_processor_task:
            self._batch_processor_task.cancel()
            try:
                await self._batch_processor_task
            except asyncio.CancelledError:
                pass
            self._batch_processor_task = None

        # Clear any pending requests
        if self._request_queue:
            while not self._request_queue.empty():
                try:
                    req = self._request_queue.get_nowait()
                    req.future.set_exception(RuntimeError("Model unloaded"))
                except asyncio.QueueEmpty:
                    break

        self._model = None
        self._tokenizer = None
        self._prompt_caches.clear()
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

    async def _batch_processor(self) -> None:
        """Background task that processes requests in batches."""
        while not self._shutdown_event.is_set():
            try:
                batch: list[PendingRequest] = []

                # Wait for first request
                try:
                    first_req = await asyncio.wait_for(
                        self._request_queue.get(),
                        timeout=1.0  # Check shutdown every second
                    )
                    batch.append(first_req)
                except asyncio.TimeoutError:
                    continue

                # Try to collect more requests for batching
                batch_deadline = time.time() + (BATCH_TIMEOUT_MS / 1000)
                while len(batch) < MAX_BATCH_SIZE:
                    remaining = batch_deadline - time.time()
                    if remaining <= 0:
                        break
                    try:
                        req = await asyncio.wait_for(
                            self._request_queue.get(),
                            timeout=remaining
                        )
                        batch.append(req)
                    except asyncio.TimeoutError:
                        break

                # Process the batch (with GPU lock to prevent concurrent access)
                if batch:
                    self._is_processing = True
                    try:
                        async with self._gpu_lock:
                            await self._process_batch(batch)
                    finally:
                        self._is_processing = False

            except asyncio.CancelledError:
                break
            except Exception as e:
                # Log error but continue processing
                print(f"Batch processor error: {e}")
                await asyncio.sleep(0.1)

    async def _process_batch(self, batch: list[PendingRequest]) -> None:
        """Process a batch of requests using batch_generate."""
        if not batch:
            return

        def _do_batch_generate():
            from mlx_lm import batch_generate
            from mlx_lm.sample_utils import make_sampler, make_repetition_penalty

            # Prepare prompts (as token lists)
            prompts = [req.prompt_tokens for req in batch]

            # Use config from first request for shared params (they should be similar)
            # In a more sophisticated implementation, we'd group by config
            config = batch[0].config

            # Create sampler
            sampler = make_sampler(
                temp=config.temperature,
                top_p=config.top_p,
            )

            # Create logits processors
            logits_processors = []
            if config.repetition_penalty > 1.0:
                logits_processors.append(
                    make_repetition_penalty(config.repetition_penalty)
                )

            # Get max_tokens for each request (may differ)
            max_tokens_list = [req.config.max_tokens for req in batch]

            try:
                result = batch_generate(
                    model=self._model,
                    tokenizer=self._tokenizer,
                    prompts=prompts,
                    max_tokens=max_tokens_list,
                    sampler=sampler,
                    logits_processors=logits_processors if logits_processors else None,
                )
                return result
            except Exception as e:
                return e

        # Run batch generation in thread pool
        result = await asyncio.to_thread(_do_batch_generate)

        # Handle errors
        if isinstance(result, Exception):
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(result)
            return

        # Distribute results to waiting requests
        # batch_generate returns BatchResponse with .responses list
        responses = result.responses if hasattr(result, 'responses') else [result]

        for req, response_text in zip(batch, responses):
            if req.future.done():
                continue

            try:
                # Get text from response
                text = response_text if isinstance(response_text, str) else str(response_text)

                # Calculate token counts
                prompt_tokens = len(req.prompt_tokens)
                completion_tokens = len(self._tokenizer.encode(text)) if self._tokenizer else len(text) // 4

                response = CompletionResponse(
                    text=text,
                    finish_reason="stop",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                )
                req.future.set_result(response)
            except Exception as e:
                req.future.set_exception(e)

    async def generate(self, request: CompletionRequest) -> CompletionResponse:
        """Generate a completion (queued for batching)."""
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        # Build prompt
        if request.messages:
            prompt_text = self._apply_chat_template(request.messages)
        elif request.prompt:
            prompt_text = request.prompt
        else:
            raise ValueError("Either prompt or messages must be provided")

        # Tokenize prompt
        prompt_tokens = self._tokenizer.encode(prompt_text)

        # Create pending request
        future: asyncio.Future[CompletionResponse] = asyncio.Future()
        pending = PendingRequest(
            id=str(uuid.uuid4()),
            prompt_tokens=prompt_tokens,
            prompt_text=prompt_text,
            config=request.config,
            future=future,
        )

        # Add to queue
        try:
            self._request_queue.put_nowait(pending)
        except asyncio.QueueFull:
            raise RuntimeError(
                f"Request queue is full ({MAX_QUEUE_SIZE} requests pending). "
                "Please try again later."
            )

        # Wait for result
        try:
            return await asyncio.wait_for(future, timeout=300)  # 5 minute timeout
        except asyncio.TimeoutError:
            raise RuntimeError("Request timed out waiting for generation")

    async def generate_stream(
        self, request: CompletionRequest
    ) -> AsyncIterator[StreamChunk]:
        """Generate a streaming completion.

        Note: Streaming currently uses single-sequence generation for simplicity.
        Batch generation returns complete responses, not streaming tokens.
        Streaming acquires the GPU lock to prevent concurrent access crashes.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        # Build prompt
        if request.messages:
            prompt = self._apply_chat_template(request.messages)
        elif request.prompt:
            prompt = request.prompt
        else:
            raise ValueError("Either prompt or messages must be provided")

        import queue
        import threading

        token_queue: queue.Queue[Optional[str]] = queue.Queue()
        stop_flag = threading.Event()

        def _stream_generate():
            from mlx_lm.generate import stream_generate
            from mlx_lm.sample_utils import make_sampler, make_repetition_penalty

            sampler = make_sampler(
                temp=request.config.temperature,
                top_p=request.config.top_p,
            )

            logits_processors = []
            if request.config.repetition_penalty > 1.0:
                logits_processors.append(
                    make_repetition_penalty(request.config.repetition_penalty)
                )

            try:
                for response in stream_generate(
                    self._model,
                    self._tokenizer,
                    prompt=prompt,
                    max_tokens=request.config.max_tokens,
                    sampler=sampler,
                    logits_processors=logits_processors if logits_processors else None,
                ):
                    if stop_flag.is_set():
                        break
                    token_queue.put(response.text)
            except Exception:
                pass  # Generation interrupted, this is fine
            finally:
                token_queue.put(None)

        # Acquire GPU lock before starting generation to prevent concurrent access
        async with self._gpu_lock:
            thread = threading.Thread(target=_stream_generate)
            thread.start()

            try:
                while True:
                    try:
                        token = await asyncio.to_thread(token_queue.get, timeout=30)
                        if token is None:
                            yield StreamChunk(text="", finish_reason="stop")
                            break
                        yield StreamChunk(text=token)
                    except queue.Empty:
                        yield StreamChunk(text="", finish_reason="timeout")
                        break
            except (asyncio.CancelledError, GeneratorExit):
                # Client disconnected - signal thread to stop and clean up
                stop_flag.set()
                raise
            finally:
                # Ensure thread is cleaned up
                stop_flag.set()
                thread.join(timeout=2)

    def get_model_info(self) -> dict:
        """Get model information."""
        info = {
            "backend": "mlx",
            "model_path": self.model_path,
            "loaded": self._loaded,
            "batching_enabled": True,
            "max_batch_size": MAX_BATCH_SIZE,
            "queue_size": self._request_queue.qsize() if self._request_queue else 0,
            "is_processing": self._is_processing,
        }

        if self._model is not None:
            try:
                if hasattr(self._model, "config"):
                    config = self._model.config
                    info["vocab_size"] = getattr(config, "vocab_size", None)
                    info["hidden_size"] = getattr(config, "hidden_size", None)
                    info["num_layers"] = getattr(config, "num_hidden_layers", None)
            except Exception:
                pass

        return info

    @property
    def queue_length(self) -> int:
        """Get current request queue length."""
        return self._request_queue.qsize() if self._request_queue else 0

    @property
    def is_queue_full(self) -> bool:
        """Check if request queue is full."""
        return self._request_queue.full() if self._request_queue else True
