#!/usr/bin/env bash
set -e

# Usage: ./docker_debug.sh [--service <service-name>] [--port <5678>] -- [args for script]
# Example: ./docker_debug.sh --service tensorrt-container --port 5678 -- --device cpu --config configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml

SERVICE=tensorrt-container
PORT=5678
# split args before/after --
while [[ "$1" != "--" && "$#" -gt 0 ]]; do
  case "$1" in
    --service) SERVICE="$2"; shift 2 ;; 
    --port) PORT="$2"; shift 2 ;; 
    *) break ;; 
  esac
done
# consume --
if [[ "$1" == "--" ]]; then shift; fi

CMD_ARGS="$@"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is not installed. Install Docker to use this helper." >&2
  exit 1
fi

# Ensure the container is up
if ! docker ps --format '{{.Names}}' | grep -q "${SERVICE}"; then
  echo "Starting container ${SERVICE} (compose up) ..."
  docker compose up -d
fi

echo "Starting debugpy in container ${SERVICE} on port ${PORT}..."
# Start the program under debugpy and wait for client
# Expose the port on the host (this assumes the container has the port accessible or uses 'docker run -p' variant)
# If using docker compose, ensure 'ports' includes ${PORT}:${PORT}

docker compose exec -T ${SERVICE} bash -lc "python -m debugpy --listen 0.0.0.0:${PORT} --wait-for-client main.py ${CMD_ARGS}"

# Connect from VS Code: attach to 127.0.0.1:${PORT}
