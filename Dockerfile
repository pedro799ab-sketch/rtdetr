
FROM nvcr.io/nvidia/pytorch:25.06-py3

WORKDIR /workspace

COPY requirements.txt .

# Build-time flag to control whether GPU-specific packages should be installed.
# Defaults to 0 (no GPU) on macOS/devcontainer builds; set to 1 on Linux GPU hosts.
ARG WITH_GPU=0

# If WITH_GPU=1, install the requirements as-is (which may include onnxruntime-gpu).
# Otherwise, remove onnxruntime-gpu from requirements and add CPU onnxruntime instead.
RUN pip install --upgrade pip && \
    if [ "${WITH_GPU}" = "1" ]; then \
        pip install -r requirements.txt; \
    else \
        grep -v '^onnxruntime-gpu' requirements.txt > /tmp/requirements-no-gpu.txt && \
        # Ensure we have a CPU onnxruntime fallback
        if ! grep -q '^onnxruntime' /tmp/requirements-no-gpu.txt; then echo 'onnxruntime' >> /tmp/requirements-no-gpu.txt; fi && \
        pip install -r /tmp/requirements-no-gpu.txt; \
    fi

CMD ["/bin/bash"]