import os
import re
from langchain.tools import tool
from tavily import TavilyClient
from dotenv import load_dotenv
import logging
from typing import Dict, List, Optional
from bs4 import BeautifulSoup, Comment, Tag
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry


load_dotenv()


# Configure structured logging for production observability
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Compile regex patterns globally at the module level to eliminate runtime recompilation overhead
NOISE_PATTERNS = re.compile(
    r"banner|cookie|share|social|comment|reply|widget|sidebar|footer|promo|ad-|^ad$|popup|modal|nav",
    re.I,
)
WHITESPACE_PATTERN = re.compile(r"\s+")

# O(1) Lookup sets for tags to speed up element validation
IGNORE_TAGS = {
    "script", "style", "noscript", "iframe", "head", "header", "footer",
    "nav", "aside", "form", "svg", "button", "modal", "object", "embed"
}
STRUCTURAL_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "blockquote"}


@tool
def web_search(query: str):
    """
    Search web page for recent and reliable information on the given query.
    """
    tavily_search = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    response = tavily_search.search(query, max_results=5)

    search_results = []
    for each_response in response["results"]:
        search_results.append(
            f"Title : {each_response.get('title')}\n"
            f"URL : {each_response.get('url')}\n"
            f"Snippet : {each_response.get('content')[:500]}\n"
        )
    return "\n----------\n".join(search_results)


def create_robust_session(
        retries: int = 3,
        backoff_factor: float = 1.5,
        status_forcelist: tuple = (429, 500, 502, 503, 504)
) -> requests.Session:
    """Configures a connection-pooled requests Session with automatic backoff and retries."""
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=["GET"],
        raise_on_status=False  # Allows us to handle status codes manually with response.raise_for_status()
    )
    adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=10)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def fetch_url_html(url: str, session: Optional[requests.Session] = None, timeout: int = 15) -> Optional[str]:
    """Fetches raw HTML from a URL with robust error handling and production headers."""
    local_session = session or create_robust_session()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    try:
        logger.info(f"Initiating HTTP GET request to: {url}")
        response = local_session.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.text
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error occurred while fetching {url}: {http_err} (Status: {response.status_code})")
    except requests.exceptions.ConnectionError as conn_err:
        logger.error(f"Network connection failed for {url}: {conn_err}")
    except requests.exceptions.Timeout as timeout_err:
        logger.error(f"Request timed out for {url} after {timeout}s: {timeout_err}")
    except requests.exceptions.RequestException as req_err:
        logger.error(f"An unexpected transport error occurred for {url}: {req_err}")

    return None


def extract_clean_content(html_content: str) -> str:
    """Extracts clean, readable text from HTML using multiple fallbacks.

    Optimized for memory efficiency, execution speed, and edge cases.
    """
    if not html_content or not html_content.strip():
        return ""

    # Specifying 'lxml' instead of 'html.parser' yields 2x-4x speed improvements in production
    soup = BeautifulSoup(html_content, "lxml")

    # 1. Clean up DOM noise efficiently
    # Decomposing inline tags within a loop mutates the tree layout. Using list extraction handles it safely.
    for tag in soup.find_all(list(IGNORE_TAGS)):
        tag.decompose()

    for comment in soup.find_all(text=lambda text: isinstance(text, Comment)):
        comment.extract()

    # Purge noise based on attributes in a single pass where possible
    for element in soup.find_all(attrs={"class": NOISE_PATTERNS, "id": NOISE_PATTERNS}):
        element.decompose()

    # 2. Strategy 1: Semantic Containment
    main_container = soup.find("article") or soup.find("main") or soup.find(id="content")
    if main_container and isinstance(main_container, Tag):
        text = _clean_element_text(main_container)
        # Using generator expression with any() to quickly validate text volume without heavy memory arrays
        if sum(1 for w in text.split()[:150]) > 100:
            logger.info("Content successfully extracted using Semantic Strategy (<article>/<main>).")
            return text

    # 3. Strategy 2: Text Density / Paragraph Clustering (Heuristic)
    # Optimized to group parent nodes using object IDs to prevent internal comparison overhead
    parent_scores: Dict[int, int] = {}
    parent_map: Dict[int, Tag] = {}

    for p in soup.find_all("p"):
        p_text = p.get_text(strip=True)
        text_len = len(p_text)
        if text_len < 25:  # Skip short noise, cookie disclaimers, etc.
            continue

        parent = p.parent
        if parent and isinstance(parent, Tag):
            p_id = id(parent)
            parent_map[p_id] = parent
            parent_scores[p_id] = parent_scores.get(p_id, 0) + text_len

    if parent_scores:
        best_parent_id = max(parent_scores, key=parent_scores.get)
        if parent_scores[best_parent_id] > 150:
            logger.info("Content successfully extracted using Paragraph Clustering Strategy.")
            return _clean_element_text(parent_map[best_parent_id])

    # 4. Strategy 3: Global Body Parsing Fallback
    if soup.body:
        logger.info("Fallback Strategy triggered: Extracting from global <body> context.")
        return _clean_element_text(soup.body)

    return _clean_element_text(soup)


def _clean_element_text(element: Tag) -> str:
    """Flattens structural nodes into readable markdown-style layout."""
    lines: List[str] = []

    # Using recursive=True with structural target sets restricts unnecessary nesting travel
    for child in element.find_all(list(STRUCTURAL_TAGS), recursive=True):
        # Prevent double-processing children of elements we already want to read
        if any(p.name in STRUCTURAL_TAGS for p in child.parents if p != element):
            continue

        text = child.get_text(strip=True)
        if not text:
            continue

        tag_name = child.name
        if tag_name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            lines.append(f"\n\n{text.upper()}\n" + "-" * min(len(text), 60))
        elif tag_name == "p":
            lines.append(f"\n{text}")
        elif tag_name == "li":
            lines.append(f"• {text}")
        elif tag_name == "blockquote":
            lines.append(f'\n\t"{text}"\n')

    # Guard clause if structural loop yields empty contents due to heavy custom UI layout configs
    if not lines:
        raw_text = element.get_text(separator="\n")
    else:
        raw_text = "\n".join(lines)

    # Sanitize whitespace efficiently across lines
    clean_lines = []
    for line in raw_text.splitlines():
        trimmed = WHITESPACE_PATTERN.sub(" ", line).strip()
        if trimmed:
            clean_lines.append(trimmed)

    result = "\n\n".join(clean_lines)
    return re.sub(r"\n\n•", "\n•", result)


@tool
def scrape_url(url):
    """Scrape the given url and provide a clean content"""
    # Initialize connection pooling
    with create_robust_session(retries=3, backoff_factor=2.0) as production_session:
        raw_html = fetch_url_html(url, session=production_session)

        if raw_html:
            cleaned_article = extract_clean_content(raw_html)

            if cleaned_article:
                return cleaned_article
            else:
                logger.warning("HTML fetched successfully but no readable content could be extracted.")
        else:
            logger.critical("Pipeline failed: Unable to fetch source data.")
