# Batch image for Cloud Run Jobs: runs one `verbatim` command and exits.
#   docker build -t verbatim .
#   docker run --rm -v "$PWD/data:/data" verbatim build-data
FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.16 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first, so this layer stays cached until uv.lock changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

# Never run as root. /data is the only writable place: on Cloud Run a Cloud
# Storage bucket is mounted over it, locally you can mount a folder.
RUN useradd --create-home --uid 1000 app && mkdir /data && chown app:app /data
ENV DATA__ROOT=/data
USER app

# The job passes the command as its args, e.g. --args=build-data
ENTRYPOINT ["verbatim"]
CMD ["check-models"]
