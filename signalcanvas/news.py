"""Lightweight web context lookup for market-moving events."""

from __future__ import annotations

from datetime import date, timedelta
from email.utils import parsedate_to_datetime
import html
import re
from typing import Any
from urllib.parse import quote_plus
from urllib.request import urlopen
import xml.etree.ElementTree as ET

ARTICLE_STOPWORDS = {
    "about", "after", "also", "amid", "and", "are", "been", "but", "for", "from", "into", "its", "last",
    "more", "near", "over", "said", "says", "stock", "stocks", "than", "that", "their", "them", "they",
    "this", "through", "with", "will", "year", "years", "while", "market", "markets", "company",
}
THEME_KEYWORDS: dict[str, set[str]] = {
    "ai demand": {"ai", "artificial", "intelligence", "gpu", "inference", "training"},
    "data center spending": {"data", "center", "cloud", "hyperscaler", "server"},
    "chip sector momentum": {"semiconductor", "chip", "chips", "sector", "peer"},
    "earnings and guidance": {"earnings", "guidance", "revenue", "profit", "outlook"},
    "analyst upgrades": {"analyst", "upgrade", "upgrades", "price", "target", "rating"},
    "product launches": {"launch", "release", "product", "blackwell", "platform"},
    "regulation and exports": {"regulation", "export", "tariff", "restriction", "policy"},
    "macro and rates": {"inflation", "rates", "fed", "treasury", "macro", "economy"},
}


def _normalize_title_and_source(title: str, source: str) -> tuple[str, str]:
    """Clean Google News titles that already embed the source name."""
    parts = [part.strip() for part in title.split(" - ") if part.strip()]
    if len(parts) >= 2 and source == "Google News":
        return " - ".join(parts[:-1]), parts[-1]
    return title, source


def _strip_html(text: str) -> str:
    plain = re.sub(r"<script[\s\S]*?</script>|<style[\s\S]*?</style>", " ", text, flags=re.IGNORECASE)
    plain = re.sub(r"<[^>]+>", " ", plain)
    plain = html.unescape(plain)
    return re.sub(r"\s+", " ", plain).strip()


def _extract_article_details(url: str) -> dict[str, str]:
    """Fetch a linked article page and extract lightweight readable fields."""
    with urlopen(url, timeout=10) as response:
        body = response.read().decode("utf-8", errors="ignore")
    title_match = re.search(r"<title[^>]*>(.*?)</title>", body, flags=re.IGNORECASE | re.DOTALL)
    description_match = re.search(
        r'<meta[^>]+(?:name="description"|property="og:description")[^>]+content="(.*?)"',
        body,
        flags=re.IGNORECASE | re.DOTALL,
    )
    paragraphs = re.findall(r"<p[^>]*>(.*?)</p>", body, flags=re.IGNORECASE | re.DOTALL)
    paragraph_text = " ".join(_strip_html(paragraph) for paragraph in paragraphs[:8])
    return {
        "page_title": _strip_html(title_match.group(1)) if title_match else "",
        "description": _strip_html(description_match.group(1)) if description_match else "",
        "excerpt": paragraph_text[:1200],
    }


def _derive_themes(articles: list[dict[str, Any]], max_themes: int = 3) -> list[str]:
    """Extract repeated finance themes from article text."""
    theme_counts = {theme: 0 for theme in THEME_KEYWORDS}
    for article in articles:
        corpus = " ".join(
            [
                article.get("title", ""),
                article.get("description", ""),
                article.get("page_title", ""),
                article.get("excerpt", ""),
            ]
        ).lower()
        tokens = set(re.findall(r"\b[a-z][a-z\-]{2,}\b", corpus))
        for theme, keywords in THEME_KEYWORDS.items():
            if len(tokens & keywords) >= 2:
                theme_counts[theme] += 1
    ranked = sorted(theme_counts.items(), key=lambda item: (-item[1], item[0]))
    return [theme for theme, count in ranked if count >= 2][:max_themes]


def lookup_event_context(symbol: str, event_date: str, move_label: str, max_items: int = 3) -> list[dict[str, Any]]:
    """Fetch nearby news headlines that may provide context for a price move."""
    target_date = date.fromisoformat(event_date)
    start_date = target_date - timedelta(days=2)
    end_date = target_date + timedelta(days=2)
    query = f'{symbol} stock "{event_date}" OR "{start_date.isoformat()}" OR "{end_date.isoformat()}" {move_label}'
    url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
    with urlopen(url, timeout=10) as response:
        xml_bytes = response.read()
    root = ET.fromstring(xml_bytes)
    items: list[dict[str, Any]] = []
    for entry in root.findall(".//item")[:max_items]:
        title = (entry.findtext("title") or "").strip()
        link = (entry.findtext("link") or "").strip()
        description = _strip_html(entry.findtext("description") or "")
        published_raw = (entry.findtext("pubDate") or "").strip()
        source_element = entry.find("{http://search.yahoo.com/mrss/}source")
        source = (source_element.text if source_element is not None else "Google News").strip()
        title, source = _normalize_title_and_source(title, source)
        if published_raw:
            try:
                published = parsedate_to_datetime(published_raw).isoformat()
            except (TypeError, ValueError):
                published = published_raw
        else:
            published = ""
        items.append(
            {
                "title": title,
                "link": link,
                "description": description,
                "published": published,
                "source": source,
            }
        )
    return [item for item in items if item["title"] and item["link"]]


def build_event_context(symbol: str, event_date: str, move_label: str, max_items: int = 3) -> dict[str, Any]:
    """Fetch headlines, enrich them with article snippets, and derive shared themes."""
    articles = lookup_event_context(symbol=symbol, event_date=event_date, move_label=move_label, max_items=max_items)
    enriched_articles: list[dict[str, Any]] = []
    for article in articles[:max_items]:
        enriched = dict(article)
        try:
            enriched.update(_extract_article_details(article["link"]))
        except Exception:
            enriched.setdefault("page_title", "")
            enriched.setdefault("excerpt", "")
        enriched_articles.append(enriched)
    return {"articles": enriched_articles, "themes": _derive_themes(enriched_articles)}
