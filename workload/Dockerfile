FROM ghcr.io/astral-sh/uv:0.11-python3.14-trixie

ENV PYTHONUNBUFFERED=1
WORKDIR /workload

RUN apt-get update && apt-get install -y curl jq

# Copy and strip generate_ledger references (local-only dev dep, not available in Docker)
COPY pyproject.toml uv.lock ./
RUN sed -i '/\[tool\.uv\.sources\]/,/^$/d; /^gen = \[/,/^\]/d' pyproject.toml

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project --no-dev

COPY . /workload
RUN sed -i '/\[tool\.uv\.sources\]/,/^$/d; /^gen = \[/,/^\]/d' pyproject.toml

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "workload.app:app", "--host", "0.0.0.0", "--port", "8000"]
