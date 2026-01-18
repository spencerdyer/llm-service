# LLM Service

A local LLM inference service with an OpenAI-compatible API. Supports Apple Silicon (MLX) and Linux (vLLM) backends with automatic platform detection.

## Features

- **OpenAI-Compatible API** - Drop-in replacement for OpenAI API endpoints
- **Multi-Platform Support** - MLX backend for Mac, vLLM backend for Linux
- **Continuous Batching** - Efficient handling of concurrent requests
- **Admin Dashboard** - Web UI for model management and monitoring
- **Model Management** - Download models from HuggingFace, configure per-model settings
- **Analytics** - Real-time metrics for requests, tokens, and generation speed
- **Streaming Support** - Server-sent events for streaming completions

## Requirements

- Python 3.11+
- [Nix](https://nixos.org/) (recommended for dependency management)
- For Mac: Apple Silicon (M1/M2/M3)
- For Linux: NVIDIA GPU with CUDA support

## Installation

### Using Nix (Recommended)

```bash
git clone <repository-url>
cd llm-service
nix develop
```

### Using pip

```bash
git clone <repository-url>
cd llm-service
pip install -e ".[mac]"  # For Mac
# or
pip install -e ".[linux]"  # For Linux
```

## Quick Start

1. Start the server:
```bash
llm-service serve
```

2. Open the admin dashboard at http://localhost:8321/admin

3. Download a model from HuggingFace (e.g., `mlx-community/Llama-3.2-3B-Instruct-4bit`)

4. Load the model and start making requests

## API Usage

The service exposes an OpenAI-compatible API at `/v1/`. Use the API key shown in the admin dashboard.

### Chat Completions

```bash
curl http://localhost:8321/v1/chat/completions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

### Streaming

```bash
curl http://localhost:8321/v1/chat/completions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": true
  }'
```

### List Models

```bash
curl http://localhost:8321/v1/models \
  -H "Authorization: Bearer YOUR_API_KEY"
```

## Configuration

Configuration can be set via environment variables (prefix: `LLM_SERVICE_`):

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_SERVICE_HOST` | Server host | `0.0.0.0` |
| `LLM_SERVICE_PORT` | Server port | `8321` |
| `LLM_SERVICE_MODELS_DIR` | Models storage directory | `~/.llm-service/models` |
| `LLM_SERVICE_API_KEY` | Override API key | Auto-generated |

## Project Structure

```
src/llm_service/
├── main.py          # FastAPI application
├── cli.py           # CLI commands
├── config.py        # Configuration settings
├── api/
│   ├── admin.py     # Admin API endpoints
│   └── openai.py    # OpenAI-compatible endpoints
├── backends/
│   ├── base.py      # Backend interface
│   ├── mlx_backend.py   # Apple Silicon backend
│   └── vllm_backend.py  # Linux/CUDA backend
├── services/
│   ├── backend_manager.py   # Model loading/inference
│   ├── config_manager.py    # Settings persistence
│   ├── model_manager.py     # Model downloads/registry
│   └── metrics.py           # Analytics tracking
└── templates/
    └── admin/
        └── dashboard.html   # Admin UI
```

## Admin Dashboard

The admin dashboard provides:

- **Status** - Current model and backend status
- **Models** - Browse, download, load, and configure models
- **Playground** - Test the model with a chat interface
- **Analytics** - Monitor request rates, token usage, and generation speed
- **Settings** - View API key and configure service settings

## Architecture

### Request Flow

1. Requests come in via the OpenAI-compatible API
2. For non-streaming requests, they're queued and batched (up to 8 requests within 50ms)
3. Batched requests are processed together for efficient GPU utilization
4. Results are distributed back to waiting clients

### Platform Detection

The service automatically detects the platform and selects the appropriate backend:
- **macOS** → MLX backend with continuous batching
- **Linux** → vLLM backend with built-in batching

### Concurrency Handling

The MLX backend includes a GPU lock to prevent Metal command buffer crashes from concurrent access. Requests are serialized at the GPU level while still benefiting from batching.

## Development

```bash
# Enter development environment
nix develop

# Run with auto-reload
llm-service serve --reload

# Run tests
pytest

# Lint
ruff check src/
```

## License

MIT
