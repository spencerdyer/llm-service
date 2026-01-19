"""CLI entry point for LLM Service."""

import os
import sys

# Force unbuffered output and proper TTY detection for progress bars
# Must be set before importing libraries that cache TTY state
os.environ.setdefault('PYTHONUNBUFFERED', '1')
if sys.stdout.isatty():
    # Ensure tqdm and huggingface_hub detect TTY properly
    os.environ.setdefault('FORCE_COLOR', '1')

import typer
from rich.console import Console

from llm_service.config import settings

app = typer.Typer(
    name="llm-service",
    help="Local LLM inference service with MLX (Mac) and vLLM (Linux) backends",
    no_args_is_help=True,
)
console = Console()


@app.command()
def serve(
    host: str = typer.Option(settings.host, "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(settings.port, "--port", "-p", help="Port to bind to"),
    reload: bool = typer.Option(settings.reload, "--reload", "-r", help="Enable auto-reload"),
):
    """Start the LLM service."""
    import uvicorn

    from llm_service.config import settings

    # Ensure directories exist
    settings.ensure_directories()

    console.print(f"[bold green]Starting LLM Service[/bold green]")
    console.print(f"  Platform: {settings.platform.value} ({settings.arch.value})")
    console.print(f"  Backend: {'MLX' if settings.is_mac else 'vLLM'}")
    console.print(f"  Data dir: {settings.data_dir}")
    console.print(f"  Models dir: {settings.effective_models_dir}")
    console.print()
    console.print(f"[bold]API:[/bold] http://{host}:{port}/v1/")
    console.print(f"[bold]Admin:[/bold] http://{host}:{port}/admin/")
    console.print()

    if settings.api_key:
        console.print(f"[dim]Using API key from environment variable[/dim]")
    else:
        console.print(f"[dim]API key will be auto-generated (view in admin panel)[/dim]")
    console.print()

    uvicorn.run(
        "llm_service.main:app",
        host=host,
        port=port,
        reload=reload,
    )


@app.command()
def info():
    """Show service information."""
    console.print("[bold]LLM Service Info[/bold]")
    console.print()
    console.print(f"  Platform: {settings.platform.value}")
    console.print(f"  Architecture: {settings.arch.value}")
    console.print(f"  Backend: {'MLX' if settings.is_mac else 'vLLM'}")
    console.print(f"  Apple Silicon: {settings.is_apple_silicon}")
    console.print()
    console.print(f"  Data directory: {settings.data_dir}")
    console.print(f"  Models directory: {settings.effective_models_dir}")
    console.print(f"  Database: {settings.effective_db_path}")
    console.print()
    if settings.api_key:
        console.print(f"  API Key: Set via environment variable")
    else:
        console.print(f"  API Key: Will be auto-generated on startup")


@app.command()
def download(
    model_id: str = typer.Argument(..., help="HuggingFace model ID (e.g., mlx-community/Llama-3-8B-4bit)"),
):
    """Download a model from HuggingFace."""
    import asyncio

    from llm_service.services.model_manager import ModelManager

    async def run():
        manager = ModelManager()
        await manager.initialize()

        console.print(f"[bold]Downloading model: {model_id}[/bold]")
        try:
            model = await manager.download_model(model_id)
            console.print(f"[green]Successfully downloaded: {model.name}[/green]")
            console.print(f"  Location: {model.local_path}")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)

    asyncio.run(run())


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
