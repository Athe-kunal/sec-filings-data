from __future__ import annotations

from typing import Literal

from mcp.server.fastmcp import FastMCP

from earnings_transcripts.transcripts import (
    get_transcript_for_quarter_async,
)
from dataloader.text_splitter import Chunk
from dataloader.vector_store import ChromaVectorStore
from mcp.server.transport_security import TransportSecuritySettings
from filings.sec_data import (
    sec_main_to_markdown_and_embed,
)
from filings.utils import company_to_ticker
from settings import sec_settings

_vector_index: ChromaVectorStore | None = None

mcp = FastMCP(
    "sec-filings-data",
    host="127.0.0.1",
    port=8000,
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=[
            "127.0.0.1:*",
            "localhost:*",
            "shirleen-supercritical-contributively.ngrok-free.dev",
        ],
        allowed_origins=[
            "http://localhost:*",
            "https://shirleen-supercritical-contributively.ngrok-free.dev",
        ],
    ),
)


def _get_vector_index() -> ChromaVectorStore:
    """Return a lazily initialized Chroma vector index client."""
    global _vector_index
    if _vector_index is None:
        _vector_index = ChromaVectorStore()
    return _vector_index


@mcp.tool()
def company_name_to_ticker_tool(name: str) -> dict[str, str]:
    """Resolve a company name to its stock ticker symbol.

    Args:
        name: Full or partial company name to resolve.
    """
    ticker = company_to_ticker(name)
    if ticker is None:
        raise ValueError(f"No ticker found for company name: {name}")
    return {"ticker": ticker}


@mcp.tool()
async def earnings_transcript_for_quarter_tool(
    ticker: str, year: int, quarter: Literal["Q1", "Q2", "Q3", "Q4"]
) -> str:
    """Fetch one earnings-call transcript and return it as markdown.

    Args:
        ticker: Equity ticker symbol, for example ``"AMZN"``.
        year: Four-digit fiscal year.
        quarter: Fiscal quarter label ``Q1``, ``Q2``, ``Q3``, or ``Q4``.
    """
    transcript = await get_transcript_for_quarter_async(ticker, year, quarter)
    if transcript is None:
        raise ValueError(
            f"No transcript available for ticker={ticker} year={year} {quarter}"
        )
    return transcript.to_markdown()


@mcp.tool()
async def sec_main_to_markdown_and_embed_tool(
    ticker: str,
    year: str,
    filing_type: str = "10-K",
    force: bool = False,
) -> dict:
    """Download one SEC filing PDF (if needed), OCR to markdown (if needed), and embed markdown."""
    payload = await sec_main_to_markdown_and_embed(
        ticker=ticker,
        year=year,
        filing_type=filing_type,
        force=force,
    )
    sec_result = payload["sec_result"]
    return {
        "sec_result": {
            "dashes_acc_num": sec_result.dashes_acc_num,
            "form_name": sec_result.form_name,
            "filing_date": sec_result.filing_date,
            "report_date": sec_result.report_date,
            "primary_document": sec_result.primary_document,
        },
        "pdf_path": str(payload["pdf_path"]),
        "markdown_path": str(payload["markdown_path"]),
        "embedded": payload["embedded"],
    }


@mcp.tool()
def search_sec_filings_tool(
    ticker: str,
    year: str,
    filing_type: str,
    query: str,
    top_k: int = 5,
) -> str:
    """Run semantic search over one indexed SEC filing.

    Args:
        ticker: Equity ticker symbol, for example ``"AMZN"``.
        year: Filing year.
        filing_type: Indexed filing key, such as ``"10-K"`` or ``"10-Q1"``.
        query: Natural-language search query.
        top_k: Maximum number of chunks to return.
    """
    results = _get_vector_index().search(
        ticker=ticker,
        year=year,
        filing_type=filing_type,
        query=query,
        top_k=top_k,
    )
    return "\n\n".join([chunk.text for chunk, _ in results])


@mcp.tool()
def search_transcripts_tool(ticker: str, year: str, query: str, top_k: int = 5) -> str:
    """Run semantic search across all indexed transcript quarters.

    Args:
        ticker: Equity ticker symbol, for example ``"AMZN"``.
        year: Transcript year.
        query: Natural-language search query.
        top_k: Maximum number of chunks to return after quarter-level merging.
    """
    year_s = str(year).strip()
    vector_index = _get_vector_index()
    resolved = vector_index.resolve_transcript_quarters(ticker, year_s)
    if not resolved:
        raise FileNotFoundError("No transcript indexes (Q1–Q4) for this ticker/year.")

    ticker_key, quarters = resolved
    merged: list[tuple[Chunk, float, str]] = []
    for filing_type in quarters:
        hits = vector_index.search(
            ticker=ticker_key,
            year=year_s,
            filing_type=filing_type,
            query=query,
            top_k=top_k,
        )
        for chunk, score in hits:
            merged.append((chunk, score, filing_type))

    merged.sort(key=lambda item: -item[1])
    merged = merged[:top_k]
    return "\n\n".join([chunk.text for chunk, _, _ in merged])


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
