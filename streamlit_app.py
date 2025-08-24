import os
import re
import io
import json
import asyncio
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional

import streamlit as st

# ------------------------- Secrets / API keys (safe lookup) -------------------------
# Avoid KeyError at import-time; read env first, then secrets (either case)
OPENAI_API_KEY = (
    os.getenv("OPENAI_API_KEY")
    or st.secrets.get("openai_api_key", st.secrets.get("OPENAI_API_KEY", ""))
)
TAVILY_API_KEY = (
    os.getenv("TAVILY_API_KEY")
    or st.secrets.get("tavily_api_key", st.secrets.get("TAVILY_API_KEY", ""))
)
YOUTUBE_API_KEY = (
    os.getenv("YOUTUBE_API_KEY")
    or st.secrets.get("youtube_api_key", st.secrets.get("YOUTUBE_API_KEY", ""))
)

# --- Windows asyncio policy fix (Playwright needs subprocess support) ---
if os.name == "nt":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except Exception:
        pass

# Optional deps
try:
    from dateutil import parser as dateparser
except Exception:
    dateparser = None

# PDF generation (prefer reportlab, fallback to fpdf)
try:
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    _HAS_RL = True
except Exception:
    _HAS_RL = False
    try:
        from fpdf import FPDF
    except Exception:
        FPDF = None

# OpenAI client (SDK v1)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ---- Import the user's scrapers ----
# YouTube (your current module name)
try:
    from youtube_scrap import get_videos_and_transcripts
except Exception:
    get_videos_and_transcripts = None

try:
    from dbta import DBTADirectScraper
except Exception as e:
    DBTADirectScraper = None
    try:
        st.error("Failed to import dbta.py — see details below.")
        st.exception(e)
    except Exception:
        pass

try:
    from sciencedaily import ScienceDailyDirectScraper
except Exception:
    ScienceDailyDirectScraper = None

try:
    from AnalyticsInsight_scrapper import AnalyticsInsightScraper
except Exception:
    AnalyticsInsightScraper = None

try:
    from tavily_scrapper import TavilyScraper
except Exception:
    TavilyScraper = None

# ------------------------- Playwright one-time init -------------------------
def ensure_playwright_chromium() -> bool:
    """
    One-time downloader for Playwright Chromium on Streamlit Cloud / fresh envs.
    Safe to call multiple times; no-ops if Chromium is already cached.
    """
    try:
        from playwright.__main__ import main as pw_main  # CLI entrypoint
        import pathlib

        cache = pathlib.Path.home() / ".cache" / "ms-playwright"
        has_chromium = cache.exists() and any(p.name.startswith("chromium") for p in cache.iterdir())
        if not has_chromium:
            pw_main(["install", "chromium"])  # downloads browser bundle
        return True
    except Exception as e:
        try:
            st.warning(f"Playwright auto-install failed: {e}")
        except Exception:
            pass
        return False

# ------------------------- Async runner (no asyncio.run) -------------------------
def run_coro(coro):
    """Run a single coroutine to completion without asyncio.run()."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        # Shouldn't happen in Streamlit, but handle just in case
        future = asyncio.ensure_future(coro, loop=loop)
        return loop.run_until_complete(future)
    else:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        finally:
            try:
                loop.close()
            finally:
                asyncio.set_event_loop(None)

# ------------------------- Date utilities -------------------------
DATE_PATTERNS = [
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+\d{1,2},\s+\d{4}\b",
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b",
    r"\b\d{4}-\d{2}-\d{2}\b",
    r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
    r"\bPosted\s+(?:[A-Za-z]+\s+\d{1,2},\s+\d{4})\b",
]

def parse_date_any(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    s = s.strip()
    s = re.sub(r"^Posted\s+", "", s, flags=re.IGNORECASE)
    try:
        if "T" in s:
            iso = s.replace("Z", "+00:00")
            return datetime.fromisoformat(iso).astimezone(timezone.utc)
    except Exception:
        pass
    if dateparser is not None:
        try:
            dt = dateparser.parse(s, fuzzy=True)
            if dt is None:
                return None
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt
        except Exception:
            pass
    m = re.search(r"(20\d{2})", s)
    if m:
        year = int(m.group(1))
        try:
            return datetime(year, 1, 1, tzinfo=timezone.utc)
        except Exception:
            return None
    return None

def extract_date_from_content(text: str) -> Optional[datetime]:
    if not text:
        return None
    around = re.search(r"(?:Published|Updated|Posted)\s*[:\-]?\s*(.+?)\b(?:\.|\n|$)", text, re.IGNORECASE)
    if around:
        dt = parse_date_any(around.group(1))
        if dt:
            return dt
    for pat in DATE_PATTERNS:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            dt = parse_date_any(m.group(0))
            if dt:
                return dt
    return None

def coerce_record(title: str, url: str, content: str, published_raw: Optional[str], fallback_text_for_date: Optional[str] = None) -> Dict[str, Any]:
    dt = parse_date_any(published_raw) if published_raw else None
    if not dt and fallback_text_for_date:
        dt = extract_date_from_content(fallback_text_for_date)
    return {
        "title": title or (url[:90] + "…" if url else "(no title)"),
        "url": url,
        "content": content or "",
        "published_raw": published_raw or "",
        "normalized_date": dt,
    }

def within_days(record: Dict[str, Any], max_age_days: int) -> bool:
    if max_age_days <= 0:
        return True
    dt = record.get("normalized_date")
    if not dt:
        return False  # drop unknown dates to guarantee recency (unless user opted in)
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    return dt >= cutoff

def dedupe_by_url(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique = []
    for r in records:
        u = (r.get("url") or "").strip()
        if u and u not in seen:
            seen.add(u)
            unique.append(r)
    return unique

# ------------------------- Summarization -------------------------
async def summarize_blocks(blocks: List[Dict[str, Any]], topic: str, model: str = "gpt-4o-mini") -> List[Dict[str, Any]]:
    if OpenAI is None:
        raise RuntimeError("openai package not installed. Please `pip install openai`.")
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set in environment or Streamlit secrets.")
    client = OpenAI(api_key=OPENAI_API_KEY)

    out = []
    for r in blocks:
        content = r.get("content", "")
        if not content:
            r["summary"] = "(No content to summarize)"
            out.append(r)
            continue
        prompt = (
            "You are a precise analyst. Summarize the following article in ~120-180 words, "
            f"focusing on takeaways related to the topic: '{topic}'. "
            "Capture key facts, dates, entities, and novel contributions. Avoid hype; be concise.\n\n"
            f"TITLE: {r.get('title','')}\nURL: {r.get('url','')}\nDATE: {r.get('normalized_date')}\n\nCONTENT:\n{content[:8000]}"
        )

        # OpenAI Chat Completions is sync over HTTP; run in executor to avoid blocking loop
        def _call_openai():
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            return resp.choices[0].message.content.strip()

        summary_text = await asyncio.to_thread(_call_openai)
        r["summary"] = summary_text
        out.append(r)
    return out

# ------------------------- PDF creation -------------------------
def make_pdf(items: List[Dict[str, Any]], topic: str) -> bytes:
    if not items:
        raise ValueError("No items to include in PDF")

    if _HAS_RL:
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=LETTER, title=f"Summaries — {topic}")
        styles = getSampleStyleSheet()
        title_style = styles["Title"]
        h_style = ParagraphStyle(name="Heading", parent=styles["Heading2"], textColor=colors.HexColor("#222222"), spaceAfter=6)
        body_style = styles["BodyText"]
        body_style.spaceAfter = 12

        story = [Paragraph(f"Summaries — {topic}", title_style), Spacer(1, 0.25 * inch)]
        for idx, it in enumerate(items, 1):
            date_str = it.get("normalized_date").strftime("%Y-%m-%d") if it.get("normalized_date") else "Unknown"
            safe_title = (it.get("title") or "Untitled").replace("&", "&amp;")
            safe_url = (it.get("url") or "").replace("&", "&amp;")
            story.append(Paragraph(f"{idx}. {safe_title}", h_style))
            story.append(Paragraph(f"<u><font color=blue>{safe_url}</font></u>", body_style))
            story.append(Paragraph(f"Date: {date_str}", body_style))
            story.append(Paragraph(it.get("summary") or "(No summary)", body_style))
            story.append(Spacer(1, 0.15 * inch))
        doc.build(story)
        return buf.getvalue()

    # Fallback using FPDF
    if FPDF is None:
        raise RuntimeError("Neither reportlab nor fpdf is available to create PDFs.")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.multi_cell(0, 10, f"Summaries — {topic}")
    pdf.ln(4)

    for idx, it in enumerate(items, 1):
        pdf.set_font("Arial", "B", 13)
        pdf.multi_cell(0, 8, f"{idx}. {it.get('title','Untitled')}")
        pdf.set_font("Arial", size=11)
        pdf.multi_cell(0, 6, f"URL: {it.get('url','')}")
        date_str = it.get("normalized_date").strftime("%Y-%m-%d") if it.get("normalized_date") else "Unknown"
        pdf.multi_cell(0, 6, f"Date: {date_str}")
        pdf.multi_cell(0, 6, it.get("summary") or "(No summary)")
        pdf.ln(2)
    return pdf.output(dest="S").encode("latin1", errors="ignore")

# ------------------------- Orchestrators -------------------------
async def run_tavily(topic: str, num_results: int) -> List[Dict[str, Any]]:
    results = []
    if TavilyScraper is None:
        st.warning("TavilyScraper module not found.")
        return results
    api_key = TAVILY_API_KEY
    if not api_key:
        st.error("TAVILY_API_KEY not set in environment or st.secrets.")
        return results
    tv = TavilyScraper(api_key, num_results=num_results)
    data = await tv.search_and_scrape(topic)
    for row in data:
        url = row.get("url", "")
        content = row.get("content", "")
        first_line = (content.split(". ")[0] if content else "").strip()[:140]
        results.append(coerce_record(title=first_line or url, url=url, content=content, published_raw=None, fallback_text_for_date=content))
    return results

async def run_dbta(topic: str, max_results: int, days_window: Optional[int]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if DBTADirectScraper is None:
        st.warning("DBTADirectScraper module not found.")
        return out
    try:
        scraper = DBTADirectScraper()
        data = await scraper.search_and_scrape(
            query=topic, days=days_window, max_results=max_results, wait_time=4
        )
    except Exception as e:
        st.error("DBTA scraping failed.")
        st.exception(e)
        return out

    if isinstance(data, list) and data and isinstance(data[0], dict) and (data[0].get("error") or data[0].get("message")):
        msg = data[0].get("error") or data[0].get("message")
        st.warning(f"DBTA: {msg}")
        return out

    for r in data or []:
        out.append(
            coerce_record(
                r.get("title"),
                r.get("url"),
                r.get("content", ""),
                r.get("published_date"),
                r.get("content", ""),
            )
        )
    return out

async def run_sciencedaily(topic: str, max_results: int, days_window: Optional[int]) -> List[Dict[str, Any]]:
    out = []
    if ScienceDailyDirectScraper is None:
        st.warning("ScienceDailyDirectScraper module not found.")
        return out
    try:
        scraper = ScienceDailyDirectScraper()
        data = await scraper.search_and_scrape(query=topic, days=days_window, max_results=max_results)
    except Exception as e:
        st.error("ScienceDaily scraping failed.")
        st.exception(e)
        return out
    for r in data or []:
        if r.get("error") or r.get("message"):
            continue
        out.append(coerce_record(r.get("title"), r.get("url"), r.get("content", ""), r.get("published_date"), r.get("content", "")))
    return out

async def run_analytics_insight(topic: str, max_results: int) -> List[Dict[str, Any]]:
    out = []
    if AnalyticsInsightScraper is None:
        st.warning("AnalyticsInsightScraper module not found.")
        return out
    # Self-heal: ensure Chromium is present (in case called directly)
    ensure_playwright_chromium()
    scraper = AnalyticsInsightScraper(query=topic)
    try:
        await scraper.scrape()
    except Exception as e:
        # Try once to install chromium then retry
        try:
            from playwright.__main__ import main as pw_main
            await asyncio.to_thread(pw_main, ["install", "chromium"])
            await scraper.scrape()
        except Exception as e2:
            st.error("Analytics Insight scraping failed.")
            st.exception(e2)
            return out
    for r in scraper.results[:max_results]:
        out.append(coerce_record(r.get("title"), r.get("link"), r.get("content", ""), r.get("date"), r.get("content", "")))
    return out

def run_youtube(topic: str, channel: str, max_results: int) -> List[Dict[str, Any]]:
    out = []
    if get_videos_and_transcripts is None:
        st.warning("youtube_scrap.py not found or failed to import.")
        return out
    try:
        data = get_videos_and_transcripts(channel_name=channel, topic=topic, max_results=max_results)
    except Exception as e:
        st.error("YouTube fetch failed.")
        st.exception(e)
        return out
    for v in data.get("videos", []):
        out.append(coerce_record(v.get("title"), v.get("url"), v.get("transcript", ""), v.get("published_at"), v.get("description", "")))
    return out

# ------------------------- Streamlit UI -------------------------
st.set_page_config(page_title="Fresh AI Research Scraper", layout="wide")

st.title("🕸️ Fresh Content Scraper → Summarizer → PDF")
st.caption("Enter a topic, fetch only the latest items, summarize with OpenAI, and export a clean PDF.")

with st.expander("🔐 Setup notes", expanded=False):
    st.markdown(
        "- Set API keys via environment variables or `st.secrets`: **OPENAI_API_KEY**, **TAVILY_API_KEY**, **YOUTUBE_API_KEY**.\n"
        "- Ensure Playwright is installed and browsers are set up (first run auto-installs Chromium).\n"
        "- Avoid hard-coding secrets in your source files."
    )

col1, col2 = st.columns([3, 2])
with col1:
    topic = st.text_input("Topic", placeholder="e.g., latest advancements in machine learning")
    days_window = st.slider("Max age (days)", min_value=1, max_value=90, value=14, help="Only keep items whose date is within this window. Items with unknown dates are dropped.")
with col2:
    max_results = st.number_input("Max results per source", 1, 25, 5)
    tavily_only = st.toggle("Tavily search only", value=False, help="When enabled, use Tavily only; otherwise run selected sources below.")

# Option to include undated items (may expand results if sites hide dates)
include_undated = st.checkbox("Include undated items (fallback)", value=False)

if not tavily_only:
    st.subheader("Sources to include")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        use_dbta = st.checkbox("DBTA", value=True)
    with c2:
        use_scidaily = st.checkbox("ScienceDaily", value=True)
    with c3:
        use_ai = st.checkbox("Analytics Insight", value=False)
    with c4:
        use_yt = st.checkbox("YouTube", value=False)
        yt_channel = st.text_input("YouTube channel (optional)", value="NeuralNine") if use_yt else ""
else:
    use_dbta = use_scidaily = use_ai = use_yt = False
    yt_channel = ""

summ_model = st.selectbox("OpenAI model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"], index=0)
run_btn = st.button("🚀 Run")

# ------------------------- collect + summarize pipeline (await-based) -------------------------
async def collect_records() -> List[Dict[str, Any]]:
    all_records: List[Dict[str, Any]] = []

    if tavily_only:
        with st.spinner("Searching via Tavily…"):
            res = await run_tavily(topic, num_results=max_results)
            all_records.extend(res)
    else:
        # If any Playwright-based source is enabled, ensure Chromium is present (first run only)
        playwright_ready = True
        need_playwright = any([use_dbta, use_scidaily, use_ai])
        if need_playwright:
            if st.session_state.get("pw_ready") is not True:
                with st.spinner("Setting up Playwright (first run downloads Chromium)…"):
                    playwright_ready = ensure_playwright_chromium()
                st.session_state["pw_ready"] = bool(playwright_ready)
            else:
                playwright_ready = True

        tasks = []
        if use_dbta and playwright_ready:
            tasks.append(run_dbta(topic, max_results=max_results, days_window=days_window))
        if use_scidaily and playwright_ready:
            tasks.append(run_sciencedaily(topic, max_results=max_results, days_window=days_window))
        if use_ai and playwright_ready:
            tasks.append(run_analytics_insight(topic, max_results=max_results))

        if need_playwright and not playwright_ready:
            st.error("Could not set up Playwright; skipping DBTA, ScienceDaily, and Analytics Insight.")

        if tasks:
            with st.spinner("Scraping async sources…"):
                results_lists = await asyncio.gather(*tasks, return_exceptions=True)
                for lst in results_lists:
                    if isinstance(lst, Exception):
                        st.exception(lst)
                    else:
                        all_records.extend(lst or [])

        if use_yt:
            with st.spinner("Fetching YouTube transcripts…"):
                all_records.extend(run_youtube(topic, yt_channel or "", max_results=max_results))

    # Dedupe and filter
    all_records = dedupe_by_url(all_records)

    filtered = []
    for r in all_records:
        if r.get("normalized_date"):
            if within_days(r, days_window):
                filtered.append(r)
        else:
            if include_undated:
                filtered.append(r)

    filtered.sort(key=lambda x: x.get("normalized_date") or datetime(1900, 1, 1, tzinfo=timezone.utc), reverse=True)
    return filtered

if run_btn:
    if not topic.strip():
        st.error("Please enter a topic.")
    elif OpenAI is None and summ_model:
        st.error("openai python package not installed. `pip install openai`.")
    elif not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY not set in environment or st.secrets.")
    else:
        # Use our custom runner to avoid asyncio.run(), but still leverage await-based tasks
        records = run_coro(collect_records())
        if not records:
            st.warning("No fresh items found within the selected window.")
        else:
            st.success(f"Found {len(records)} fresh item(s). Summarizing…")
            try:
                summarized = run_coro(summarize_blocks(records, topic=topic, model=summ_model))
            except Exception as e:
                st.exception(e)
                summarized = []

            if summarized:
                show = []
                for r in summarized:
                    show.append({
                        "date": r.get("normalized_date").strftime("%Y-%m-%d") if r.get("normalized_date") else "Unknown",
                        "title": r.get("title"),
                        "url": r.get("url"),
                        "chars": len(r.get("content") or ""),
                    })
                st.dataframe(show, use_container_width=True, hide_index=True)

                try:
                    pdf_bytes = make_pdf(summarized, topic)
                    st.download_button(
                        "⬇️ Download PDF",
                        data=pdf_bytes,
                        file_name=f"summaries_{re.sub(r'[^a-zA-Z0-9]+','_', topic.lower())}.pdf",
                        mime="application/pdf",
                    )
                    st.success("PDF generated.")
                except Exception as e:
                    st.exception(e)
            else:
                st.warning("Nothing to export.")

st.markdown("---")
st.caption(
    "This app uses flexible date parsing and content scanning to keep only the latest items. "
    "Unknown-dated items are dropped by design (unless you opt in)."
)
