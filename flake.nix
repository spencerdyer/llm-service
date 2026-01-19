{
  description = "LLM Service - Local LLM inference with MLX (Mac) and vLLM (Linux)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
          overlays = [
            (final: prev: {
              python311 = prev.python311.override {
                packageOverrides = pyFinal: pyPrev: {
                  # Disable tests for packages that fail in Nix sandbox due to
                  # AF_UNIX socket path length limits in long build paths
                  pydantic = pyPrev.pydantic.overridePythonAttrs (old: {
                    doCheck = false;
                  });
                  fsspec = pyPrev.fsspec.overridePythonAttrs (old: {
                    doCheck = false;
                  });
                };
              };
            })
          ];
        };

        # Platform detection
        isDarwin = pkgs.stdenv.isDarwin;
        isLinux = pkgs.stdenv.isLinux;
        isAarch64 = pkgs.stdenv.hostPlatform.isAarch64;

        # Platform-specific Python packages
        pythonPkgs = pkgs.python311.withPackages (ps: with ps; [
          # Core dependencies
          fastapi
          uvicorn
          pydantic
          pydantic-settings
          httpx
          aiofiles
          jinja2
          python-multipart

          # Database
          sqlalchemy
          aiosqlite

          # HuggingFace
          huggingface-hub

          # Utilities
          rich
          typer
          pyyaml
        ]);

        # Common environment variables
        commonEnv = {
          LLM_SERVICE_PLATFORM = if isDarwin then "darwin" else "linux";
          LLM_SERVICE_ARCH = if isAarch64 then "aarch64" else "x86_64";
        };

      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pythonPkgs
            pkgs.git
          ] ++ pkgs.lib.optionals isDarwin [
            # Mac-specific: we'll install mlx-lm via pip in the shell
          ] ++ pkgs.lib.optionals isLinux [
            # Linux-specific: C++ stdlib required by vLLM and other native extensions
            pkgs.stdenv.cc.cc.lib
          ];

          shellHook = ''
            export LLM_SERVICE_PLATFORM="${commonEnv.LLM_SERVICE_PLATFORM}"
            export LLM_SERVICE_ARCH="${commonEnv.LLM_SERVICE_ARCH}"
            export LLM_SERVICE_DATA_DIR="''${LLM_SERVICE_DATA_DIR:-$PWD/data}"
            export LLM_SERVICE_MODELS_DIR="''${LLM_SERVICE_MODELS_DIR:-$PWD/data/models}"

            # Add C++ stdlib and system CUDA to library path (Linux only)
            # Note: We must NOT add system glibc paths to avoid conflicts with Nix glibc
            ${pkgs.lib.optionalString isLinux ''
              export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:''${LD_LIBRARY_PATH:-}"
              export CUDA_HOME="/usr/local/cuda"
              export PATH="/usr/local/cuda/bin:$PATH"

              # Add CUDA library paths (runtime libs are in targets/x86_64-linux/lib)
              for cuda_lib in /usr/local/cuda/lib64 /usr/local/cuda/targets/x86_64-linux/lib; do
                if [ -d "$cuda_lib" ]; then
                  export LD_LIBRARY_PATH="$cuda_lib:$LD_LIBRARY_PATH"
                fi
              done

              # Create a local directory with symlinks to just the NVIDIA driver CUDA libs
              # This avoids adding system glibc paths which would conflict with Nix
              nvidia_stub_dir="$PWD/.nix-nvidia-libs"
              mkdir -p "$nvidia_stub_dir" 2>/dev/null || true
              for lib in /lib/x86_64-linux-gnu/libcuda.so* /lib/x86_64-linux-gnu/libnvidia*.so*; do
                if [ -f "$lib" ]; then
                  ln -sf "$lib" "$nvidia_stub_dir/" 2>/dev/null || true
                fi
              done
              export LD_LIBRARY_PATH="$nvidia_stub_dir:$LD_LIBRARY_PATH"
            ''}

            echo "LLM Service Development Environment"
            echo "===================================="
            echo "Platform: $LLM_SERVICE_PLATFORM ($LLM_SERVICE_ARCH)"
            echo "Data directory: $LLM_SERVICE_DATA_DIR"
            echo "Models directory: $LLM_SERVICE_MODELS_DIR"
            echo ""

            # Create data directories
            mkdir -p "$LLM_SERVICE_DATA_DIR"
            mkdir -p "$LLM_SERVICE_MODELS_DIR"

            # Create virtual environment for platform-specific packages
            if [ ! -d .venv ]; then
              echo "Creating virtual environment for platform-specific packages..."
              python -m venv .venv
            fi

            source .venv/bin/activate

            # Install platform-specific ML backends via pip
            # (nixpkgs vLLM has broken dependencies, so we use pip for ML backends)
            if [ "$LLM_SERVICE_PLATFORM" = "darwin" ]; then
              echo "Installing MLX packages for Mac..."
              pip install -q mlx mlx-lm 2>/dev/null || echo "Note: Install mlx-lm manually if needed"
            else
              echo "Installing vLLM for Linux..."
              pip install -q vllm 2>/dev/null || echo "Note: Install vllm manually if needed"
            fi

            # Install the project in editable mode
            pip install -q -e . 2>/dev/null || true

            echo ""
            echo "Commands:"
            echo "  llm-service serve    - Start the LLM service"
            echo "  llm-service --help   - Show all commands"
            echo ""
          '';
        };

        packages.default = pkgs.writeShellScriptBin "llm-service" ''
          export LLM_SERVICE_PLATFORM="${commonEnv.LLM_SERVICE_PLATFORM}"
          export LLM_SERVICE_ARCH="${commonEnv.LLM_SERVICE_ARCH}"
          exec ${pythonPkgs}/bin/python -m llm_service "$@"
        '';

        apps.default = {
          type = "app";
          program = "${self.packages.${system}.default}/bin/llm-service";
        };

        apps.serve = {
          type = "app";
          program = toString (pkgs.writeShellScript "llm-service-serve" ''
            export LLM_SERVICE_PLATFORM="${commonEnv.LLM_SERVICE_PLATFORM}"
            export LLM_SERVICE_ARCH="${commonEnv.LLM_SERVICE_ARCH}"
            cd ${self}
            exec ${pythonPkgs}/bin/python -m llm_service serve "$@"
          '');
        };
      }
    );
}
