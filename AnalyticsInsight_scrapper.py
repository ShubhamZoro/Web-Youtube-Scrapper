import json
import asyncio
from datetime import datetime
from playwright.async_api import async_playwright

class AnalyticsInsightScraper:
    def __init__(self, query="ML"):
        self.base_url = "https://www.analyticsinsight.net/search?q="
        self.query = query
        self.results = []

    async def scrape(self):
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            # Step 1: Open search page
            search_url = f"{self.base_url}{self.query}"
            await page.goto(search_url, timeout=60000)
            await page.wait_for_selector('[data-test-id="headline"] a')

            # Step 2: Extract article links
            links = await page.eval_on_selector_all(
                '[data-test-id="headline"] a',
                "elements => elements.map(el => el.href)"
            )

            print(f"Found {len(links)} articles")

            # Step 3: Visit each article page
            for link in links:
                try:
                    article_page = await browser.new_page()
                    await article_page.goto(link, timeout=60000)

                    # Extract title
                    title = await article_page.title()

                    # Extract published date if available
                    try:
                        date = await article_page.locator("time").first.text_content()
                    except:
                        date = "Unknown"

                    # Extract main content (p tags)
                    paragraphs = await article_page.eval_on_selector_all(
                        "p", "elements => elements.map(el => el.innerText)"
                    )
                    content = "\n".join(paragraphs)

                    self.results.append({
                        "title": title,
                        "link": link,
                        "date": date,
                        "content": content
                    })
                    await article_page.close()

                except Exception as e:
                    print(f"Error scraping {link}: {e}")

            await browser.close()

    def save_to_json(self, filename="analyticsinsight_articles.json"):
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=4)
        print(f"Saved {len(self.results)} articles to {filename}")


# async def main():
#     scraper = AnalyticsInsightScraper(query="ML")
#     await scraper.scrape()
#     scraper.save_to_json()
