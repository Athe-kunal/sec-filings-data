import json
import re
import asyncio
import requests
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from loguru import logger

from filings import utils
from settings import sec_settings


def sec_data_case_dir(ticker: str, year: str) -> Path:
    """Directory for one ticker/year: ``{sec_data_dir}/{ticker}-{year}/``."""
    return Path(sec_settings.sec_data_dir) / f"{ticker}-{year}"


@dataclass(frozen=True)
class SecResults:
    dashes_acc_num: str
    form_name: str
    filing_date: str
    report_date: str
    primary_document: str


def _parse_filing_type_for_sec_query(
    filing_type: str,
) -> tuple[frozenset[int] | None, frozenset[str]]:
    """Build SEC ``form`` names and an optional 10-Q quarter filter (1–3 from ``10-Qn``).

    Plain ``10-Q`` keeps all quarters; ``10-Q4`` raises (fourth quarter is ``10-K``).
    """
    raw = filing_type.strip()
    u = raw.upper().replace(" ", "")
    m = re.fullmatch(r"10-Q(\d)", u)
    if m:
        q = int(m.group(1))
        if q not in (1, 2, 3):
            if q == 4:
                raise ValueError(
                    "10-Q4 is not a valid filing type; the fourth quarter is filed as 10-K."
                )
            raise ValueError(
                f"Invalid quarterly type {raw!r}; use 10-Q1, 10-Q2, or 10-Q3."
            )
        return frozenset({q}), frozenset({"10-Q"})
    if u == "10-Q":
        return None, frozenset({"10-Q"})
    return None, frozenset({raw})


def get_sec_results(
    ticker: str,
    year: str,
    filing_type: str = "10-K",
    company: str | None = None,
    email: str | None = None,
) -> list[SecResults]:
    """Fetch SEC filing metadata for the given ticker and year.

    Pass ``10-K``, plain ``10-Q`` (all ``10-Q`` filings for the year), or
    ``10-Q1`` / ``10-Q2`` / ``10-Q3`` to restrict by fiscal quarter from
    ``reportDate``. Only non-amended ``10-Q`` rows are included; ``10-Q4`` is not
    allowed (use ``10-K``).
    """
    company = company or sec_settings.sec_api_organization
    email = email or sec_settings.sec_api_email
    cik = utils.get_cik_by_ticker(ticker)
    logger.info(f"For {ticker=} found {cik=}")

    quarter_filter, forms = _parse_filing_type_for_sec_query(filing_type)

    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    headers = {
        "User-Agent": f"{company} {email}",
        "Content-Type": "text/html",
    }
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        json_data = response.json()
    else:
        raise RuntimeError(
            f"Unable to fetch submissions. Status code: {response.status_code}"
        )

    filings = json_data["filings"]
    recent_filings = filings["recent"]
    sec_form_names: list[str] = []
    form_lists: list[SecResults] = []

    for acc_num, form_name, filing_date, report_date, primary_doc in zip(
        recent_filings["accessionNumber"],
        recent_filings["form"],
        recent_filings["filingDate"],
        recent_filings["reportDate"],
        recent_filings["primaryDocument"],
        strict=True,
    ):
        if form_name in forms and report_date.startswith(str(year)):
            display_name = form_name
            if form_name == "10-Q":
                datetime_obj = datetime.strptime(report_date, "%Y-%m-%d")
                quarter = (datetime_obj.month + 2) // 3
                if quarter_filter is not None and quarter not in quarter_filter:
                    continue
                display_name = f"10-Q{quarter}"
                if display_name in sec_form_names:
                    display_name += "-1"
            no_dashes_acc_num = re.sub("-", "", acc_num)
            form_lists.append(
                SecResults(
                    dashes_acc_num=no_dashes_acc_num,
                    form_name=display_name,
                    filing_date=filing_date,
                    report_date=report_date,
                    primary_document=primary_doc,
                )
            )
            sec_form_names.append(display_name)

    return form_lists


async def save_sec_results_as_pdfs(
    sec_results: list[SecResults],
    ticker: str,
    year: str,
    company: str | None = None,
    email: str | None = None,
) -> list[Path]:
    """Save SEC results as PDF files and persist metadata to ``sec_results.json``."""
    company = company or sec_settings.sec_api_organization
    email = email or sec_settings.sec_api_email
    cik = utils.get_cik_by_ticker(ticker)
    rgld_cik = int(cik.lstrip("0"))
    output_dir = sec_data_case_dir(ticker, year)
    output_dir.mkdir(parents=True, exist_ok=True)

    filings_to_save = [
        utils.FilingToSave(
            cik=rgld_cik,
            accession_number=sr.dashes_acc_num,
            primary_document=sr.primary_document,
            output_path=output_dir / f"{sr.form_name}.pdf",
        )
        for sr in sec_results
    ]

    pdf_paths = await utils.save_filings_as_pdfs(
        filings=filings_to_save,
        company=company,
        email=email,
    )

    # Persist metadata so filing dates are available without re-hitting SEC API.
    json_path = output_dir / sec_settings.sec_metadata_filename
    json_path.write_text(
        json.dumps([asdict(sr) for sr in sec_results], indent=2),
        encoding="utf-8",
    )

    logger.info(f"Saved {len(pdf_paths)} PDFs and metadata to {output_dir}")
    return pdf_paths


def load_sec_results(ticker: str, year: str) -> list[SecResults]:
    """Load previously persisted SEC filing metadata from ``sec_results.json``.

    Returns an empty list if the file does not yet exist (i.e. filings have
    not been downloaded yet for this ticker/year).
    """
    json_path = sec_data_case_dir(ticker, year) / sec_settings.sec_metadata_filename
    if not json_path.exists():
        return []
    records = json.loads(json_path.read_text(encoding="utf-8"))
    return [SecResults(**r) for r in records]


async def sec_main(
    ticker: str,
    year: str,
    filing_type: str = "10-K",
) -> tuple[list[SecResults], list[Path]]:
    """Fetch SEC results and save them as PDFs.

    Use ``10-Q1``, ``10-Q2``, or ``10-Q3`` to fetch one quarter (from report date);
    ``10-Q`` fetches all quarterly filings for the year.
    """
    ticker_name = utils.company_to_ticker(ticker)
    assert ticker_name, f"The {ticker=} that you provided, is not valid"
    sec_results = get_sec_results(
        ticker=ticker,
        year=year,
        filing_type=filing_type,
    )
    pdf_paths = await save_sec_results_as_pdfs(
        sec_results=sec_results,
        ticker=ticker,
        year=year,
    )
    return sec_results, pdf_paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch SEC filings and save as PDFs")
    parser.add_argument("--ticker", type=str, required=True, help="Stock ticker symbol")
    parser.add_argument("--year", type=str, required=True, help="Filing year")
    parser.add_argument(
        "--filing-type",
        type=str,
        default="10-K",
        help="SEC form to fetch (e.g. 10-K, 10-Q, 10-Q1, 10-Q2, 10-Q3)",
    )
    args = parser.parse_args()

    asyncio.run(
        sec_main(
            ticker=args.ticker,
            year=args.year,
            filing_type=args.filing_type,
        )
    )
