import os
import json
import asyncio
import requests
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
import streamlit as st

class TavilyScraper:
    def __init__(self, api_key: str, num_results: int = 5):
        self.api_key = api_key
        self.num_results = num_results

    # -------------------- Tavily Search --------------------
    def tavily_search(self, query: str):
        url = "https://api.tavily.com/search"
        headers = {"Content-Type": "application/json"}
        payload = {
            "api_key": self.api_key,
            "query": query,
            "num_results": self.num_results
        }

        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

        return [item["url"] for item in data.get("results", [])]

    # -------------------- Playwright Scraper --------------------
    async def scrape_page(self, url: str) -> str:
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context()
                page = await context.new_page()

                await page.goto(url, timeout=30000)
                html = await page.content()

                await browser.close()

                soup = BeautifulSoup(html, "html.parser")

                # Remove unwanted tags
                for script in soup(["script", "style", "noscript"]):
                    script.extract()

                text = " ".join(soup.stripped_strings)
                return text  # limit size
        except Exception as e:
            return f"Error scraping {url}: {str(e)}"

    # -------------------- Search + Scrape --------------------
    async def search_and_scrape(self, query: str):
        urls = self.tavily_search(query)
        results = []

        for url in urls:
            content = await self.scrape_page(url)
            results.append({"url": url, "content": content})

        return results


# -------------------- Run Example --------------------
if __name__ == "__main__":
    TAVILY_API_KEY = st.secrets['tavily_api_key']

    query = "latest advancements in machine learning"

    scraper = TavilyScraper(TAVILY_API_KEY, num_results=3)

    async def run():
        data = await scraper.search_and_scrape(query)
        with open("scraped_results.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print("✅ Scraping completed. Results saved to scraped_results.json")

    asyncio.run(run())
