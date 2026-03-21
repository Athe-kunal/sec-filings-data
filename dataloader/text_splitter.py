import re
from dataclasses import dataclass

import markdownify
from langchain_text_splitters import RecursiveCharacterTextSplitter

_PAGE_BLOCK_RE = re.compile(
    r"<PAGE-NUM-(\d+)>\s*([\s\S]*?)\s*</PAGE-NUM-\1>",
    re.IGNORECASE,
)
_PAGE_TAG_RE = re.compile(r"</?PAGE-NUM-\d+>", re.IGNORECASE)
_TABLE_RE = re.compile(r"<table[\s\S]*?</table>", re.IGNORECASE)
_SECTION_TITLE_RE = re.compile(
    r"^(Item\s+\d+[A-C]?\..*|Part\s+[IV]+.*)",
    re.MULTILINE | re.IGNORECASE,
)
_SEC_CHUNK_SIZE = 1024
_SEC_CHUNK_OVERLAP = 128
_EARNINGS_TRANSCRIPT_CHUNK_SIZE = 2048
_EARNINGS_TRANSCRIPT_OVERLAP = 256


@dataclass
class Chunk:
    """A single logical unit extracted from a markdown document."""

    text: str
    chunk_type: str  # "table" | "text"
    page_num: int | None
    section_title: str | None
    index: int


def _extract_section(text: str, current_section: str | None) -> str | None:
    title_match = _SECTION_TITLE_RE.search(text)
    if title_match:
        return title_match.group(0).strip()
    return current_section


def _build_splitter(chunk_size: int, overlap: int) -> RecursiveCharacterTextSplitter:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=False,
        strip_whitespace=True,
    )


def _extract_pages(markdown_text: str) -> list[tuple[int | None, str]]:
    matches = list(_PAGE_BLOCK_RE.finditer(markdown_text))
    if not matches:
        cleaned = _PAGE_TAG_RE.sub("", markdown_text).strip()
        return [(None, cleaned)] if cleaned else []

    pages: list[tuple[int | None, str]] = []
    for match in matches:
        page_num = int(match.group(1))
        page_text = match.group(2).strip()
        if page_text:
            pages.append((page_num, page_text))
    return pages


def _split_tables(page_text: str) -> tuple[list[str], list[str]]:
    return _TABLE_RE.split(page_text), _TABLE_RE.findall(page_text)


def chunk_markdown(
    text: str,
    *,
    chunk_size: int = _SEC_CHUNK_SIZE,
    overlap: int = _SEC_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Chunk SEC markdown by page using langchain recursive splitting.

    Tables are extracted and emitted as standalone markdown table chunks.
    Remaining text content is split per-page with overlap.
    """
    splitter = _build_splitter(chunk_size=chunk_size, overlap=overlap)
    pages = _extract_pages(text)

    chunks: list[Chunk] = []
    index = 0
    current_section: str | None = None

    for page_num, page_text in pages:
        non_table_parts, tables = _split_tables(page_text)

        for part_idx, part in enumerate(non_table_parts):
            clean_text = _PAGE_TAG_RE.sub("", part).strip()
            if clean_text:
                current_section = _extract_section(clean_text, current_section)
                for split_text in splitter.split_text(clean_text):
                    chunks.append(
                        Chunk(
                            text=split_text,
                            chunk_type="text",
                            page_num=page_num,
                            section_title=current_section,
                            index=index,
                        )
                    )
                    index += 1

            if part_idx < len(tables):
                table_markdown = markdownify.markdownify(
                    tables[part_idx].strip(), heading_style="ATX"
                ).strip()
                if table_markdown:
                    chunks.append(
                        Chunk(
                            text=table_markdown,
                            chunk_type="table",
                            page_num=page_num,
                            section_title=current_section,
                            index=index,
                        )
                    )
                    index += 1

    return chunks


def chunk_transcript_rows(
    rows: list[tuple[str, str]],
    chunk_size: int = _EARNINGS_TRANSCRIPT_CHUNK_SIZE,
    overlap: int = _EARNINGS_TRANSCRIPT_OVERLAP,
) -> list[Chunk]:
    """Create transcript chunks using RecursiveCharacterTextSplitter overlap."""
    splitter = _build_splitter(chunk_size=chunk_size, overlap=overlap)

    chunks: list[Chunk] = []
    index = 0

    for speaker, text in rows:
        clean_speaker = speaker.strip()
        clean_text = text.strip()
        if not clean_text:
            continue

        for part in splitter.split_text(clean_text):
            chunk_text = f"Speaker: {clean_speaker}\nText: {part}" if clean_speaker else part
            chunks.append(
                Chunk(
                    text=chunk_text,
                    chunk_type="text",
                    page_num=None,
                    section_title=clean_speaker or None,
                    index=index,
                )
            )
            index += 1

    return chunks
