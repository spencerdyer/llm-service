"""Configuration management for LLM Service."""

import os
import secrets
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def generate_api_key() -> str:
    """Generate a secure random API key."""
    return f"llm-{secrets.token_urlsafe(32)}"


class Platform(str, Enum):
    DARWIN = "darwin"
    LINUX = "linux"


class Architecture(str, Enum):
    AARCH64 = "aarch64"
    X86_64 = "x86_64"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="LLM_SERVICE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Platform detection (set by Nix or auto-detected)
    platform: Platform = Field(
        default_factory=lambda: Platform.DARWIN if os.uname().sysname == "Darwin" else Platform.LINUX
    )
    arch: Architecture = Field(
        default_factory=lambda: Architecture.AARCH64 if os.uname().machine == "arm64" else Architecture.X86_64
    )

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8321
    reload: bool = False

    # API settings
    api_key: Optional[str] = None  # Required for /v1/* endpoints

    # Data directories
    data_dir: Path = Field(default_factory=lambda: Path.cwd() / "data")
    models_dir: Optional[Path] = None  # Defaults to data_dir/models

    # Database
    db_path: Optional[Path] = None  # Defaults to data_dir/llm_service.db

    # Model settings
    default_model: Optional[str] = None
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9  # For vLLM

    @property
    def effective_models_dir(self) -> Path:
        """Get the effective models directory."""
        if self.models_dir:
            return self.models_dir
        return self.data_dir / "models"

    @property
    def effective_db_path(self) -> Path:
        """Get the effective database path."""
        if self.db_path:
            return self.db_path
        return self.data_dir / "llm_service.db"

    @property
    def is_mac(self) -> bool:
        """Check if running on Mac."""
        return self.platform == Platform.DARWIN

    @property
    def is_linux(self) -> bool:
        """Check if running on Linux."""
        return self.platform == Platform.LINUX

    @property
    def is_apple_silicon(self) -> bool:
        """Check if running on Apple Silicon."""
        return self.is_mac and self.arch == Architecture.AARCH64

    def ensure_directories(self) -> None:
        """Create necessary directories."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.effective_models_dir.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
