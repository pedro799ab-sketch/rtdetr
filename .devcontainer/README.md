# Devcontainer notes

This workspace contains multiple docker-compose files to handle different platforms.

- `docker-compose.yml` — the main, full-featured compose file (may include NVIDIA-specific settings and deploy resources used on Linux with GPUs).
- `.devcontainer/docker-compose.mac.yml` — simplified compose for macOS / Docker Desktop (no NVIDIA runtime or device reservations).

If you're on macOS, `devcontainer.json` is configured to use the mac compose file so ``Reopen in Container`` should work.

If the container still fails to start, see the troubleshooting commands in the root README and gather the Dev Containers log (View → Output → Dev Containers) and the output of:

```bash
docker info
docker compose -f docker-compose.yml -f .devcontainer/docker-compose.mac.yml config
docker compose -f docker-compose.yml -f .devcontainer/docker-compose.mac.yml up -d tensorrt-container
docker compose -f docker-compose.yml -f .devcontainer/docker-compose.mac.yml logs -f tensorrt-container
```

Paste those outputs into an issue or share them so we can diagnose further.

Switching configs
------------------
If you prefer to use the Dockerfile-based devcontainer (recommended on macOS), the repo's main `devcontainer.json` now uses the Dockerfile at the repo root. The devcontainer build is configured to target `linux/amd64` (QEMU emulation) to improve availability of binary wheels like `onnxruntime-gpu` on macOS. The original compose-based configuration is saved as `.devcontainer/devcontainer.compose.json` and can be restored by copying it over `devcontainer.json`.

Example to restore compose-based config:
GPU package control
-------------------
The Dockerfile supports a `WITH_GPU` build argument to control whether GPU-only packages (like `onnxruntime-gpu`) are installed.

- Default (used by the devcontainer): `WITH_GPU=0` — installs CPU `onnxruntime` instead of `onnxruntime-gpu`.
- To enable GPU packages (on a Linux machine with proper GPU drivers), build with `WITH_GPU=1`.

Examples:

```bash
# Build locally with GPU packages enabled (only on Linux GPU hosts)
docker buildx build --platform linux/amd64 --build-arg WITH_GPU=1 -f Dockerfile -t rtdert:gpu ..

# For Dev Containers, set the build arg in .devcontainer/devcontainer.json:
# "build": { "args": { "WITH_GPU": "1" } }
```


```bash
cp .devcontainer/devcontainer.compose.json .devcontainer/devcontainer.json
```

