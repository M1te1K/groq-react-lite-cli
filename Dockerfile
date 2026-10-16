FROM ghcr.io/astral-sh/uv:0.13.1 AS uv
FROM python:3.12-slim
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable && useradd --create-home agent && mkdir logs && chown agent:agent logs
USER agent
ENTRYPOINT ["/app/.venv/bin/react-lite"]
