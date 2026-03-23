FROM ghcr.io/astral-sh/uv:0.10-python3.13-trixie

ENV PYTHONUNBUFFERED=1
WORKDIR /workload

RUN apt-get update && apt-get install -y curl jq

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

COPY . /workload

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "workload.app:app", "--host", "0.0.0.0", "--port", "8000"]
