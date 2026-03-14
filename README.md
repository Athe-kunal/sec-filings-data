# olmocr-sec-filings

## Configuration

Settings are loaded via Pydantic Settings from environment variables or a `.env` file:

| Variable | Description | Default |
|----------|-------------|---------|
| `SEC_API_ORGANIZATION` | Organization name for SEC API User-Agent | `Your-Organization` |
| `SEC_API_EMAIL` | Contact email for SEC API User-Agent | `your-email@example.com` |
| `OLMOCR_SERVER` | vLLM server URL for olmOCR | `http://localhost:8000/v1` |
| `OLMOCR_MODEL` | Model name for olmOCR | `allenai/olmOCR-2-7B-1025-FP8` |
| `OLMOCR_WORKSPACE` | Workspace directory for OCR output | `./localworkspace` |

Import the singleton `settings` instance:

```python
from settings import settings

company = settings.sec_api_organization
email = settings.sec_api_email
server = settings.olmocr_server
```

## Docker

### Build

```bash
docker build -t sec-filings-md .
```

### Run

```bash
docker run --gpus device=${CUDA_VISIBLE_DEVICES:-3} \
  -e SEC_API_ORGANIZATION="Your-Organization" \
  -e SEC_API_EMAIL="your-email@example.com" \
  -e OLMOCR_SERVER="http://host.docker.internal:8000/v1" \
  -e CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3} \
  -v ./sec_data:/app/sec_data \
  -v ./localworkspace:/app/localworkspace \
  -p 8081:8081 \
  sec-filings-md
```

The two volumes persist data across container restarts:

| Volume | Container path | Purpose |
|--------|---------------|---------|
| `sec_data` | `/app/sec_data` | Downloaded SEC filing PDFs |
| `localworkspace` | `/app/localworkspace` | OCR workspace and output markdown |

Override the workspace path at runtime with `-e OLMOCR_WORKSPACE=/custom/path`.

## Installation

```bash
uv sync
playwright install chromium
```

## Usage

Start vLLM server:
```bash
make vllm-olmocr-serve
```

Fetch SEC filings:
```bash
uv run python -m filings.sec_data --ticker AMZN --year 2025
```

Run OCR pipeline:
```bash
uv run python ocr/olmocr_pipeline.py --pdf-dir sec_data/AMZN-2025
```
