# olmocr-sec-filings

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
