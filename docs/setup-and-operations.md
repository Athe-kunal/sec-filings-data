# Setup and Operations Guide

This guide complements `README.md` and focuses on day-to-day setup and runtime commands.

## 1) Read README.md first

Start with the root `README.md` for project context, environment variables, and architecture notes.

## 2) Installation

Use `uv` for dependency management.

```bash
uv sync
```

For OCR + embeddings workflows:

```bash
uv sync --group ocr-md
```

For MCP workflows:

```bash
uv sync --group ocr-md --group mcp
```

## 3) Docker setup

Use the Makefile targets (recommended):

```bash
make docker-build
make docker-run
```

Or run both with one command:

```bash
make docker-start
```

Useful lifecycle commands:

```bash
make docker-stop
make docker-remove
```

Makefile variables you can override:

- `IMAGE_NAME`
- `GPU_DEVICE`
- `API_PORT`
- `SEC_API_ORGANIZATION`
- `SEC_API_EMAIL`

Example:

```bash
make docker-start GPU_DEVICE=0 API_PORT=8081
```

## 4) Start model servers

Use Makefile commands so ports and options stay consistent.

### olmOCR vLLM server

```bash
make vllm-olmocr-serve
```

### Embedding vLLM server (Qwen pooling)

```bash
make vllm-embd-serve
```

> Requirement: Use a GPU with **at least 12 GB VRAM** for local olmOCR + Qwen embedding serving.

Optional benchmark command:

```bash
make guidellm-benchmark
```

## 5) Start API server

```bash
make start-server
```

This runs Uvicorn for `server:app` on `0.0.0.0` using `API_PORT`.

## 6) MCP server and ngrok

Install MCP dependencies:

```bash
uv sync --group ocr-md --group mcp
```

Start MCP server:

```bash
uv run --group ocr-md --group mcp python mcp_server.py
```

Expose MCP with ngrok (replace `8069` if your MCP port differs):

```bash
ngrok http 8069
```

Then:

1. Copy the public ngrok hostname.
2. Add it to `MCP_NGROK_ALLOWED_HOSTS` in `.env` as a JSON array.
3. Restart `mcp_server.py`.
4. Connect your MCP client to `https://<ngrok-hostname>/mcp`.

Example `.env` value:

```bash
MCP_NGROK_ALLOWED_HOSTS='["example-name.ngrok-free.app"]'
```

## 7) Makefile command reference

- `make vllm-olmocr-serve`: Start olmOCR model server.
- `make vllm-embd-serve`: Start embedding model server.
- `make start-server`: Start FastAPI service.
- `make guidellm-benchmark`: Run throughput benchmark.
- `make docker-build`: Build Docker image.
- `make docker-run`: Run Docker container.
- `make docker-start`: Build and run Docker.
- `make docker-stop`: Stop container exposing `API_PORT`.
- `make docker-remove`: Remove container exposing `API_PORT`.
