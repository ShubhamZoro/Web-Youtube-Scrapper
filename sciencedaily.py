import os
import re
import asyncio
import json
import sys
import platform
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup

# Optional dependencies with guards
try:
    from playwright.async_api import async_playwright
    _HAS_PLAYWRIGHT = True
except Exception:
    _HAS_PLAYWRIGHT = False


class ScienceDailyDirectScraper:
    """
    Direct ScienceDaily website scraper that:
    1. Opens sciencedaily.com
    2. Uses the search form (#keyword input)
    3. Extracts links from .gsc-webResult.gsc-result .gs-title a
    4. Scrapes individual articles and checks timeframes
    5. Filters results based on publication dates
    
    Usage:
        scraper = ScienceDailyDirectScraper()
        results = await scraper.search_and_scrape("machine learning", days=7)
        scraper.save_to_json(results, "sciencedaily_results.json")
    """

    def __init__(self, use_playwright: bool = True):
        self.use_playwright = use_playwright and _HAS_PLAYWRIGHT
        self.base_url = "https://www.sciencedaily.com"

        # Date formats for parsing
        self.date_formats = [
            "%Y-%m-%d",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%SZ",
            "%B %d, %Y",      # August 15, 2025
            "%b %d, %Y",      # Aug 15, 2025
            "%d %B %Y",       # 15 August 2025
            "%d %b %Y",       # 15 Aug 2025
            "Posted %B %d, %Y",  # Posted August 15, 2025
        ]

    def _parse_date_flexible(self, date_str: str) -> Optional[datetime]:
        """Parse date from string with multiple format attempts"""
        if not date_str:
            return None
            
        date_str = date_str.strip()
        
        # Clean up common prefixes
        date_str = re.sub(r'^Posted\s+', '', date_str, flags=re.IGNORECASE)
        
        # Try ISO format first
        try:
            if "T" in date_str:
                iso = date_str.replace("Z", "+00:00")
                return datetime.fromisoformat(iso).astimezone(timezone.utc)
        except Exception:
            pass
            
        # Try other formats
        for fmt in self.date_formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                else:
                    dt = dt.astimezone(timezone.utc)
                return dt
            except Exception:
                continue
                
        return None

    def _is_within_timeframe(self, published_date: str, days: Optional[int], reference_date: Optional[datetime] = None) -> bool:
        """Check if content is within specified timeframe"""
        if not days or not published_date:
            return True
            
        ref_date = reference_date or datetime.now(timezone.utc)
        cutoff_date = ref_date - timedelta(days=days)
        
        parsed_date = self._parse_date_flexible(published_date)
        if not parsed_date:
            return True  # Include if can't parse date
            
        return parsed_date >= cutoff_date

    def _extract_dates_from_soup(self, soup: BeautifulSoup) -> List[str]:
        """Extract potential date strings from HTML soup"""
        dates = []
        
        # Look for meta tags with date information
        meta_selectors = [
            ("meta", {"itemprop": "datePublished"}),
            ("meta", {"property": "article:published_time"}),
            ("meta", {"name": "article:published_time"}),
            ("meta", {"property": "og:updated_time"}),
            ("meta", {"name": "pubdate"}),
            ("meta", {"name": "date"}),
        ]
        
        for selector in meta_selectors:
            elem = soup.find(*selector)
            if elem and elem.get("content"):
                dates.append(elem["content"].strip())

        # Look for time elements
        for time_elem in soup.find_all("time"):
            for attr in ("datetime", "pubdate"):
                value = time_elem.get(attr)
                if value:
                    dates.append(value.strip())
            text = time_elem.get_text(" ", strip=True)
            if text:
                dates.append(text)

        # Look for common date patterns in text
        date_patterns = [
            r'Posted\s+([A-Za-z]+ \d{1,2}, \d{4})',
            r'Published:?\s*([A-Za-z]+ \d{1,2}, \d{4})',
            r'Date:?\s*([A-Za-z]+ \d{1,2}, \d{4})',
            r'\b([A-Za-z]+ \d{1,2}, \d{4})\b',
            r'\b(\d{4}-\d{2}-\d{2})\b'
        ]
        
        page_text = soup.get_text()
        for pattern in date_patterns:
            matches = re.findall(pattern, page_text)
            dates.extend(matches)

        # Remove duplicates while preserving order
        return list(dict.fromkeys(dates))

    def _clean_content(self, content: str) -> str:
        """Clean and normalize scraped content"""
        if not content:
            return ""
            
        # Normalize whitespace
        content = re.sub(r"\n\s*\n", "\n\n", content)
        content = re.sub(r"\s+", " ", content)
        
        # Remove unwanted patterns
        unwanted_patterns = [
            r"Advertisement",
            r"Sponsored content", 
            r"Subscribe.*?newsletter",
            r"Follow us.*?social",
            r"Cookie.*?policy",
            r"Privacy.*?policy",
            r"Terms.*?service",
            r"Sign in.*?continue",
            r"Log in.*?account",
            r"Accept.*?cookies",
            r"Share this.*?article"
        ]
        
        for pattern in unwanted_patterns:
            content = re.sub(pattern, "", content, flags=re.IGNORECASE)
            
        return content.strip()

    async def search_and_scrape(
        self,
        query: str,
        days: Optional[int] = None,
        max_results: int = 10,
        wait_time: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Search ScienceDaily website and scrape results with time filtering
        
        Args:
            query: Search query to enter in the search box
            days: Number of days to look back (optional)
            max_results: Maximum number of results to return
            wait_time: Time to wait after search (seconds)
            
        Returns:
            List of scraped articles with metadata
        """
        
        if not self.use_playwright:
            return [{"error": "Playwright is required but not available"}]

        reference_date = datetime.now(timezone.utc)
        results = []

        print(f"🔍 Searching ScienceDaily for: '{query}'")
        if days:
            print(f"⏰ Time filter: Last {days} day(s)")

        try:
            async with async_playwright() as p:
                # Windows-specific browser args
                browser_args = [
                    "--no-sandbox",
                    "--disable-setuid-sandbox", 
                    "--disable-dev-shm-usage",
                    "--disable-gpu",
                ]
                
                # Add Windows-specific args
                if platform.system() == "Windows":
                    browser_args.extend([
                        "--disable-features=VizDisplayCompositor",
                        "--disable-background-timer-throttling",
                        "--disable-renderer-backgrounding",
                        "--disable-backgrounding-occluded-windows"
                    ])
                
                browser = await p.chromium.launch(
                    headless=True,
                    args=browser_args,
                )
                page = await browser.new_page()
                
                await page.set_extra_http_headers({
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                })

                # Navigate to ScienceDaily
                print("🌐 Opening ScienceDaily website...")
                await page.goto(self.base_url, wait_until="domcontentloaded", timeout=60000)
                await page.wait_for_timeout(2000)
                
                # Handle any popups/cookies
                try:
                    popup_selectors = [
                        'button:has-text("Accept")',
                        'button:has-text("OK")',
                        'button:has-text("Got it")',
                        'button:has-text("Allow")',
                        'button:has-text("Continue")',
                        '[id*="cookie"] button',
                        '[class*="cookie"] button'
                    ]
                    
                    for selector in popup_selectors:
                        try:
                            await page.click(selector, timeout=1500)
                            await page.wait_for_timeout(500)
                            break
                        except Exception:
                            pass
                except Exception:
                    pass

                # Perform search
                print(f"🔎 Performing search for: {query}")
                try:
                    await page.fill("#keyword", query)
                    await page.keyboard.press("Enter")
                    await page.wait_for_timeout(wait_time * 1000)
                    
                    print(f"⏳ Waiting {wait_time} seconds for search results...")
                    
                except Exception as e:
                    print(f"❌ Error with search: {e}")
                    await browser.close()
                    return [{"error": f"Search error: {e}"}]

                # Extract search result links
                print("📄 Extracting article links from search results...")
                
                try:
                    # Wait for search results to load
                    await page.wait_for_selector(".gsc-webResult.gsc-result", timeout=10000)
                    
                    # Collect result links
                    article_links = await page.eval_on_selector_all(
                        ".gsc-webResult.gsc-result .gs-title a",
                        "els => els.map(el => ({href: el.href, title: el.textContent.trim()}))"
                    )

                    # Remove duplicates
                    seen = set()
                    unique_links = []
                    for link in article_links:
                        url = link.get("href")
                        title = link.get("title", "")
                        if url and url not in seen:
                            seen.add(url)
                            unique_links.append({"url": url, "title": title})

                    print(f"🔗 Found {len(unique_links)} unique result links")
                    
                except Exception as e:
                    print(f"❌ Error extracting search results: {e}")
                    await browser.close()
                    return [{"error": f"Results extraction error: {e}"}]

                # Process each article
                for i, link in enumerate(unique_links[:max_results], 1):
                    url = link["url"]
                    title = link["title"]
                    print(f"📝 Processing article {i}/{min(len(unique_links), max_results)}: {title[:50]}...")

                    try:
                        # Navigate to the article
                        await page.goto(url, wait_until="domcontentloaded", timeout=60000)
                        await page.wait_for_timeout(1500)

                        # Get page content
                        html_content = await page.content()
                        soup = BeautifulSoup(html_content, "html.parser")

                        # Extract dates
                        extracted_dates = self._extract_dates_from_soup(soup)

                        # Check time filtering
                        if days and extracted_dates:
                            is_within_timeframe = any(
                                self._is_within_timeframe(date, days, reference_date)
                                for date in extracted_dates
                            )
                            if not is_within_timeframe:
                                print(f"⏭️ Skipping (outside time range): {title[:50]}...")
                                continue

                        # Extract main content
                        try:
                            text = await page.evaluate("""
                                () => {
                                    const removeSelectors = [
                                        'script', 'style', 'nav', 'header', 'footer', 'aside',
                                        '.ad', '.ads', '.advertisement', '.sidebar', '.comments',
                                        '.social-share', '.related-posts', '.newsletter-signup',
                                        '[role="banner"]', '[role="navigation"]', '[role="complementary"]',
                                        '.menu', '.navigation', '.breadcrumb', '.tags', '.categories'
                                    ];
                                    
                                    removeSelectors.forEach(sel => {
                                        document.querySelectorAll(sel).forEach(el => el.remove());
                                    });
                                    
                                    const mainSelectors = [
                                        'article', 'main', '[role="main"]',
                                        '.content', '.post-content', '.article-content',
                                        '.entry-content', '.page-content', '.main-content',
                                        '.story-body', '.article-body', '.content-body'
                                    ];
                                    
                                    let mainElement = null;
                                    for (const sel of mainSelectors) {
                                        mainElement = document.querySelector(sel);
                                        if (mainElement && mainElement.innerText.length > 100) break;
                                    }
                                    
                                    if (!mainElement) {
                                        mainElement = document.body;
                                    }
                                    
                                    return (mainElement.innerText || mainElement.textContent || '').trim();
                                }
                            """)
                            
                            text = " ".join((text or "").split())
                        except Exception:
                            text = soup.get_text()
                            
                        # Clean content
                        cleaned_content = self._clean_content(text)

                        # Get final title
                        final_title = title
                        if not final_title:
                            h1 = soup.find("h1")
                            if h1:
                                final_title = h1.get_text(strip=True)

                        # Best publication date
                        best_date = extracted_dates[0] if extracted_dates else ""

                        # Create result
                        result = {
                            "title": final_title or url,
                            "url": url,
                            "content": cleaned_content,
                            "published_date": best_date,
                            "domain": "sciencedaily.com",
                            "query": query,
                            "content_source": "playwright_direct",
                            "content_length": len(cleaned_content),
                            "word_count": len(cleaned_content.split()) if cleaned_content else 0,
                            "extracted_dates": extracted_dates,
                            "scraped_at": datetime.now().isoformat(),
                            "days_filter": days,
                            "search_method": "direct_website"
                        }
                        
                        results.append(result)
                        print(f"✅ Added: {len(cleaned_content)} chars, {result['word_count']} words")
                        print(f"📅 Publication date: {best_date}")

                        # Rate limiting
                        await page.wait_for_timeout(800)

                    except Exception as e:
                        print(f"❌ Error processing {url}: {e}")
                        continue

                await browser.close()

        except Exception as e:
            print(f"❌ Browser error: {e}")
            return [{"error": f"Browser error: {e}"}]

        if not results:
            print("🤷 No results found matching criteria")
            return [{"message": "No results found matching the specified criteria"}]
            
        print(f"🎉 Completed! Total results: {len(results)}")
        return results

    def save_to_json(self, results: List[Dict[str, Any]], filename: str = "sciencedaily_results.json") -> None:
        """Save results to JSON with comprehensive metadata"""
        
        if not results:
            print("❌ No results to save")
            return
            
        # Handle error/message results
        if len(results) == 1 and (results[0].get("error") or results[0].get("message")):
            output_data = {
                "scraping_metadata": {
                    "scraped_at": datetime.now().isoformat(),
                    "total_results": 0,
                    "status": results[0].get("error") or results[0].get("message"),
                    "method": "direct_website_search",
                    "platform": platform.system()
                },
                "scraped_content": []
            }
        else:
            # Calculate statistics
            total_content = sum(r.get("content_length", 0) for r in results)
            total_words = sum(r.get("word_count", 0) for r in results)
            queries = list(set(r.get("query", "") for r in results if r.get("query")))
            dates_found = [r.get("published_date", "") for r in results if r.get("published_date")]
            
            output_data = {
                "scraping_metadata": {
                    "scraped_at": datetime.now().isoformat(),
                    "total_results": len(results),
                    "total_content_length": total_content,
                    "total_word_count": total_words,
                    "average_content_length": total_content // max(1, len(results)),
                    "average_word_count": total_words // max(1, len(results)),
                    "queries_used": queries,
                    "publication_dates_found": len([d for d in dates_found if d]),
                    "method": "direct_website_search",
                    "domain": "sciencedaily.com",
                    "platform": platform.system(),
                    "python_version": sys.version
                },
                "scraped_content": results
            }

        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
                
            print(f"💾 Results saved to: {filename}")
            
            if output_data["scraping_metadata"]["total_results"] > 0:
                meta = output_data["scraping_metadata"]
                print(f"📊 Summary:")
                print(f"   • Total results: {meta['total_results']}")
                print(f"   • Total content: {meta['total_content_length']:,} characters")
                print(f"   • Total words: {meta['total_word_count']:,}")
                print(f"   • Articles with dates: {meta['publication_dates_found']}")
                print(f"   • Average per article: {meta['average_content_length']:,} chars")
                print(f"   • Platform: {meta['platform']}")
                
        except Exception as e:
            print(f"❌ Error saving results: {e}")


# Example usage
async def main():
    """Demonstration of direct ScienceDaily scraping"""
    
    scraper = ScienceDailyDirectScraper()
    
    print("🚀 Starting Direct ScienceDaily Website Scraping")
    print(f"🖥️  Platform: {platform.system()} {platform.release()}")
    print(f"🐍 Python: {sys.version.split()[0]}")
    print()
    
    # Example: Search for AI articles from past 7 days
    print("=" * 60)
    print("🤖 Searching for Artificial Intelligence Articles (Past 7 days)")
    print("=" * 60)
    results = await scraper.search_and_scrape(
        query="Artificial Intelligence", 
        days=7, 
        max_results=5,
        wait_time=4
    )
    
    print("\n" + "=" * 60)
    print("💾 Saving Results")
    print("=" * 60)
    scraper.save_to_json(results, "sciencedaily_direct_scraping_results.json")


def run_main():
    """Windows-compatible way to run async main function"""
    if platform.system() == "Windows":
        # Set the event loop policy for Windows
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    try:
        # Try to get existing event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If already running (e.g., in Jupyter), create a task
            import nest_asyncio
            nest_asyncio.apply()
            asyncio.run(main())
        else:
            # Run normally
            loop.run_until_complete(main())
    except RuntimeError:
        # If no event loop exists, create and run
        asyncio.run(main())


if __name__ == "__main__":
    run_main()
