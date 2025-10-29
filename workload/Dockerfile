FROM ghcr.io/astral-sh/uv:0.9-python3.13-trixie

ENV PYTHONUNBUFFERED=1
WORKDIR /workload
RUN apt-get update && apt-get install -y curl vim jq
# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Ensure installed tools can be executed out of the box
ENV UV_TOOL_BIN_DIR=/usr/local/bin

# Install dependencies in a separate layer
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

# Copy the project into the image
COPY . /workload

# Sync the project
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev

# COPY test_composer/all_transactions /opt/antithesis/test/v1/all_transactions
EXPOSE 8000
# CMD ["workload"]
ENV PATH="/workload/.venv/bin:$PATH"
CMD ["tail", "-f", "/dev/null"]
# CMD ["uvicorn", "workload.app:app", "--host", "0.0.0.0", "--port", "8000",
#      "--reload", "--reload-dir", "/opt/antithesis/catalog/workload/src/workload"]
