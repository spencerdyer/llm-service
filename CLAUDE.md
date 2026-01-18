# LLM Service - Claude Code Context

## Environment

This project uses **Nix** for dependency management. You must first enter the nix shell before running commands:

```bash
cd <project-root>  # Navigate to this project's directory
nix develop
```

Only after entering the nix shell can you run:
- `llm-service serve` - Start the server
- `python -m llm_service` - Alternative way to run

## Project Structure

- `src/llm_service/` - Main Python package
  - `main.py` - FastAPI app entry point
  - `cli.py` - Typer CLI commands
  - `config.py` - Pydantic settings (env prefix: `LLM_SERVICE_`)
  - `api/` - API routes (admin, openai)
  - `services/` - Business logic (model_manager, config_manager, backend_manager, metrics)
  - `backends/` - MLX (Mac) and vLLM (Linux) inference backends with batching support
  - `templates/` - Jinja2 HTML templates for admin UI
  - `static/` - Static assets
- `flake.nix` - Nix flake configuration
- `pyproject.toml` - Python project config

## Key Configuration

- Default port: 8321
- Models directory: configurable via admin panel or `LLM_SERVICE_MODELS_DIR`
- Database: SQLite at `data/llm_service.db`
- Environment variable prefix: `LLM_SERVICE_`

## Running the Server

From inside `nix develop`:
```bash
llm-service serve                    # Default port 8321
llm-service serve --port 9000        # Custom port
llm-service serve --reload           # Dev mode with auto-reload
```

## API Endpoints

### OpenAI-Compatible API (requires Bearer token auth)
- `POST /v1/chat/completions` - Chat completions
- `POST /v1/completions` - Text completions
- `GET /v1/models` - List available models

### Admin API (no auth required for local use)
- `GET /admin/` - Admin dashboard UI
- `GET /api/admin/status` - Backend status
- `GET /api/admin/models` - List all models
- `POST /api/admin/models/download` - Download model from HuggingFace
- `POST /api/admin/models/{id}/load` - Load a model
- `GET /api/admin/metrics/*` - Analytics endpoints

## Architecture Notes

- **MLX Backend** (Mac): Uses continuous batching via `mlx-lm` for concurrent request handling
- **vLLM Backend** (Linux): Uses vLLM's built-in batching
- **Request Queue**: Requests are batched (up to 8) within 50ms windows for efficient GPU utilization
- **GPU Lock**: Ensures only one operation accesses Metal/GPU at a time to prevent crashes
