"""FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from llm_service import __version__
from llm_service.api import admin, openai
from llm_service.config import settings
from llm_service.services.backend_manager import BackendManager
from llm_service.services.config_manager import ConfigManager
from llm_service.services.metrics import MetricsService
from llm_service.services.model_manager import ModelManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Initialize services
    settings.ensure_directories()

    app.state.config_manager = ConfigManager()
    await app.state.config_manager.initialize()

    app.state.model_manager = ModelManager(config_manager=app.state.config_manager)
    await app.state.model_manager.initialize()

    app.state.backend_manager = BackendManager(
        model_manager=app.state.model_manager,
        config_manager=app.state.config_manager,
    )

    app.state.metrics_service = MetricsService()
    await app.state.metrics_service.initialize()

    # Initialize API key
    from llm_service.config import generate_api_key

    # Priority: environment variable > stored in DB > generate new
    if settings.api_key:
        # Use environment variable if set
        app.state.api_key = settings.api_key
        print(f"Using API key from environment variable")
    else:
        # Check if we have a stored API key
        stored_key = await app.state.config_manager.get_setting("api_key")
        if stored_key:
            app.state.api_key = stored_key
            print(f"Using stored API key")
        else:
            # Generate a new API key
            new_key = generate_api_key()
            await app.state.config_manager.set_setting("api_key", new_key)
            app.state.api_key = new_key
            print(f"Generated new API key: {new_key}")

    # Load default model if configured
    default_model = await app.state.config_manager.get_setting("default_model")
    if default_model:
        try:
            await app.state.backend_manager.load_model(default_model)
        except Exception as e:
            print(f"Warning: Could not load default model '{default_model}': {e}")

    yield

    # Cleanup
    await app.state.metrics_service.close()
    await app.state.backend_manager.shutdown()


app = FastAPI(
    title="LLM Service",
    description="Local LLM inference service with MLX (Mac) and vLLM (Linux) backends",
    version=__version__,
    lifespan=lifespan,
)

# Include routers
app.include_router(openai.router, prefix="/v1", tags=["OpenAI API"])
app.include_router(admin.router, prefix="/admin", tags=["Admin"])
app.include_router(admin.api_router, prefix="/api/admin", tags=["Admin API"])

# Mount static files and templates
import importlib.resources as pkg_resources
from pathlib import Path

# Get the package directory
package_dir = Path(__file__).parent
templates_dir = package_dir / "templates"
static_dir = package_dir / "static"

if templates_dir.exists():
    templates = Jinja2Templates(directory=str(templates_dir))
    app.state.templates = templates

if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def root():
    """Redirect root to admin panel."""
    return RedirectResponse(url="/admin/")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": __version__,
        "platform": settings.platform.value,
        "backend": "mlx" if settings.is_mac else "vllm",
    }
