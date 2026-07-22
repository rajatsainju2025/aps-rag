"""
the logbook page fetcher for the Studies Copilot.

Validates that a URL matches the allowed the logbook pattern:
  https://your-host.example.com/entry/views/itemDomainLogbook/view?id=<ENTRY_ID>
  where ENTRY_ID is a sequence of digits (e.g. 10427, 123, 1237172).

Fetches the page, strips HTML tags, and returns clean plain text.
Only this URL pattern is permitted — any other URL is rejected.
"""
import logging
import re
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)

# ─── Allowlist ────────────────────────────────────────────────────────────────
_ENTRY_PATTERN = re.compile(
    r"^https://your-host\.example\.com/entry/views/itemDomainLogbook/view\?id=(\d+)$"
)

# Regex to find the system URLs anywhere in a text string (for extraction from queries)
_ENTRY_URL_FINDER = re.compile(
    r"https://your-host\.example\.com/entry/views/itemDomainLogbook/view\?id=(\d+)"
)


def is_entry_url(url: str) -> bool:
    """Return True iff the URL matches the the logbook allowlist pattern."""
    return bool(_ENTRY_PATTERN.match(url.strip()))


def extract_entry_urls(text: str) -> List[Tuple[str, str]]:
    """Extract all the logbook URLs from a text string.

    Returns a list of (url, entry_id) tuples in order of appearance.
    """
    matches = _ENTRY_URL_FINDER.findall(text)
    results = []
    for entry_id in matches:
        url = f"https://your-host.example.com/entry/views/itemDomainLogbook/view?id={entry_id}"
        results.append((url, entry_id))
    return results


def fetch_entry_page(url: str, timeout: int = 15) -> Optional[str]:
    """Fetch and parse a the logbook page.

    Args:
        url: Must match the the system allowlist pattern exactly.
        timeout: HTTP request timeout in seconds.

    Returns:
        Cleaned plain-text content of the page, or None on failure.

    Raises:
        ValueError: If the URL does not match the the system allowlist pattern.
    """
    url = url.strip()
    if not is_entry_url(url):
        raise ValueError(
            f"URL not allowed: '{url}'. Only the logbook URLs matching "
            "https://your-host.example.com/entry/views/itemDomainLogbook/view?id=<digits> "
            "are permitted."
        )

    try:
        import httpx
    except ImportError:
        logger.error("httpx is required for the system page fetching (pip install httpx)")
        return None

    logger.info("Fetching the system page: %s", url)
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()
            html = response.text
    except httpx.TimeoutException:
        logger.warning("Timeout fetching the system page: %s", url)
        return None
    except httpx.HTTPStatusError as e:
        logger.warning("HTTP error fetching the system page %s: %s", url, e)
        return None
    except Exception as e:
        logger.warning("Error fetching the system page %s: %s", url, e)
        return None

    return _parse_html(html, url)


def _parse_html(html: str, url: str) -> str:
    """Extract meaningful text from the logbook page HTML.

    Tries BeautifulSoup first for clean extraction; falls back to
    regex-based stripping if bs4 is not installed.
    """
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")

        # Remove script / style noise
        for tag in soup(["script", "style", "nav", "footer", "head"]):
            tag.decompose()

        # Prefer the main content area if identifiable
        main = (
            soup.find("main")
            or soup.find("div", class_=re.compile(r"content|entry|logbook|item", re.I))
            or soup.body
        )
        text = (main or soup).get_text(separator="\n")

    except ImportError:
        # Fallback: strip all HTML tags with regex
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"&[a-z]+;", " ", text)

    # Normalise whitespace
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # Deduplicate consecutive identical lines (navigation items repeat)
    deduped = []
    for ln in lines:
        if not deduped or ln != deduped[-1]:
            deduped.append(ln)

    result = "\n".join(deduped)
    entry_id = _ENTRY_PATTERN.match(url).group(1) if _ENTRY_PATTERN.match(url) else "?"
    logger.info("the system page id=%s parsed: %d chars", entry_id, len(result))
    return result


def format_entry_attachment(url: str, entry_id: str, content: str) -> str:
    """Format fetched the system page content as a labelled attachment block."""
    return (
        f"<entry_logbook_entry id=\"{entry_id}\" url=\"{url}\">\n"
        f"{content[:6000]}"
        + ("\n[... content truncated ...]" if len(content) > 6000 else "")
        + f"\n</entry_logbook_entry>"
    )
