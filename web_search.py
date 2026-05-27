"""
Web Search Module
- Uses ddgs library (pip install ddgs)
- Falls back to manual lite.duckduckgo.com scrape
- Works offline-first: if search fails, says so clearly
"""

import re
import time
import urllib.request
import urllib.parse


# ─── Primary search: ddgs library ─────────────────────────────

def search_duckduckgo(query, max_results=5):
    """
    Search the web and return list of results.
    Each result: {'title': str, 'url': str, 'snippet': str}
    Returns empty list on failure.
    """
    results = _search_with_ddgs(query, max_results)
    if results:
        return results

    # Fallback: scrape lite.duckduckgo.com
    print("[SEARCH] Library failed, trying lite fallback...")
    return _search_lite_fallback(query, max_results)


def _search_with_ddgs(query, max_results):
    """Primary: use the ddgs metasearch library."""
    try:
        from ddgs import DDGS

        ddgs = DDGS(timeout=10)
        raw  = ddgs.text(query, max_results=max_results)

        results = []
        for r in raw:
            results.append({
                "title":   r.get("title", ""),
                "url":     r.get("href", ""),
                "snippet": r.get("body", ""),
            })
        return results

    except ImportError:
        print("[SEARCH] ddgs not installed. Run: pip install ddgs")
        return []
    except Exception as e:
        err = str(e).lower()
        if "ratelimit" in err:
            print("[SEARCH] Rate limited by search engine. Waiting 5s...")
            time.sleep(5)
        else:
            print(f"[SEARCH] DDGS error: {e}")
        return []


# ─── Fallback: scrape lite.duckduckgo.com ─────────────────────

def _search_lite_fallback(query, max_results):
    """Fallback: POST to lite.duckduckgo.com and parse HTML."""
    try:
        url  = "https://lite.duckduckgo.com/lite/"
        data = urllib.parse.urlencode({"q": query}).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/125.0.0.0 Safari/537.36"
                ),
                "Content-Type":  "application/x-www-form-urlencoded",
                "Referer":       "https://lite.duckduckgo.com/",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "same-origin",
                "Sec-Fetch-User": "?1",
            },
        )

        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="ignore")

        return _parse_lite_html(html, max_results)

    except Exception as e:
        print(f"[SEARCH] Lite fallback error: {e}")
        return []


def _parse_lite_html(html, max_results):
    """Parse the DDG Lite results page."""
    results = []

    links = re.findall(
        r'<a[^>]+class="result-link"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
        html, re.DOTALL,
    )
    snippets = re.findall(
        r'<td[^>]+class="result-snippet"[^>]*>(.*?)</td>',
        html, re.DOTALL,
    )

    for i, (url, title) in enumerate(links[:max_results]):
        title   = re.sub(r"<[^>]+>", "", title).strip()
        url     = url.strip()
        snippet = ""
        if i < len(snippets):
            snippet = re.sub(r"<[^>]+>", "", snippets[i]).strip()

        if title and url:
            results.append({
                "title":   title,
                "url":     url,
                "snippet": snippet,
            })

    return results


# ─── Formatting helpers ───────────────────────────────────────

def format_results_for_ai(query, results):
    """Format search results into a prompt for the AI to summarize."""
    if not results:
        return f"No search results found for: {query}"

    lines = [f"Search query: {query}\n", "Results:\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['title']}")
        lines.append(f"   {r['url']}")
        lines.append(f"   {r['snippet']}")
        lines.append("")
    return "\n".join(lines)


def format_results_for_speech(query, results):
    """Format results for voice / short summary."""
    if not results:
        return f"I could not find any results for {query}."

    lines = [f"Based on search results for '{query}', here is what I found:\n"]
    for r in results[:3]:
        lines.append(f"{r['title']}: {r['snippet']}")
    return "\n".join(lines)


# ─── Quick test ───────────────────────────────────────────────

if __name__ == "__main__":
    query = "latest AI news 2026"
    print(f"Searching: {query}\n")
    results = search_duckduckgo(query)
    if results:
        for r in results:
            print(f"  {r['title']}")
            print(f"  {r['url']}")
            print(f"  {r['snippet'][:120]}...")
            print()
    else:
        print("No results found.")