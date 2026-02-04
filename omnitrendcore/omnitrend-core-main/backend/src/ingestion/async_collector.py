import asyncio
import aiohttp
import logging
import feedparser
import ssl
import traceback
from datetime import datetime
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
from urllib.parse import urlparse

from .config import Config, SourceConfig, CollectionConfig
from .robots_checker import RobotsChecker
from .collectors.base import Article

logger = logging.getLogger(__name__)

class AsyncNewsCollector:
    """Asynchronous orchestrator for news collection"""
    
    def __init__(self, collection_config=None):
        self.collection_config = collection_config or Config.DEFAULT_COLLECTION
        self.robots_checker = RobotsChecker(
            user_agent=self.collection_config.user_agent,
            cache_dir=Config.CACHE_DIR
        )
        self.all_articles: List[Article] = []
        self.domain_locks: Dict[str, asyncio.Lock] = {}
        self.last_fetch_time: Dict[str, float] = {}
        
        # SSL context to bypass verification (matching the fix applied to sequential collector)
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE

    def _get_domain(self, url: str) -> str:
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}"

    async def _async_respect_rate_limit(self, url: str):
        """Async-friendly rate limiting per domain"""
        domain = self._get_domain(url)
        if domain not in self.domain_locks:
            self.domain_locks[domain] = asyncio.Lock()
        
        async with self.domain_locks[domain]:
            # Get delay from robots.txt or config
            delay = self.collection_config.request_delay
            try:
                loop = asyncio.get_running_loop()
                robots_delay = await loop.run_in_executor(None, self.robots_checker.get_crawl_delay, url)
                if robots_delay:
                    delay = max(delay, robots_delay)
            except:
                pass
            
            now = asyncio.get_event_loop().time()
            last_time = self.last_fetch_time.get(domain, 0)
            wait_time = delay - (now - last_time)
            
            if wait_time > 0:
                logger.debug(f"Async waiting {wait_time:.2f}s for domain {domain}")
                await asyncio.sleep(wait_time)
            
            self.last_fetch_time[domain] = asyncio.get_event_loop().time()

    async def fetch_url(self, session: aiohttp.ClientSession, url: str) -> Optional[str]:
        """Fetch URL content with rate limiting"""
        await self._async_respect_rate_limit(url)
        
        try:
            timeout = aiohttp.ClientTimeout(total=60, connect=30)
            async with session.get(url, ssl=self.ssl_context, timeout=timeout) as response:
                if response.status == 200:
                    # Check content type
                    content_type = response.headers.get('Content-Type', '').lower()
                    if 'application/pdf' in content_type or 'image/' in content_type:
                        logger.debug(f"Skipping non-text content: {url} ({content_type})")
                        return None
                        
                    # Use replace to avoid encoding errors
                    return await response.text(errors='replace')
                else:
                    logger.warning(f"Failed to fetch {url}: Status {response.status}")
        except Exception as e:
            logger.error(f"Error fetching {url}: {repr(e)}")
            logger.debug(traceback.format_exc())
        return None

    async def collect_from_rss(self, session: aiohttp.ClientSession, source: SourceConfig) -> List[Article]:
        """Asynchronously collect from an RSS source"""
        logger.info(f"Starting async RSS collection from {source.name}")
        content = await self.fetch_url(session, source.url)
        if not content:
            return []
        
        feed = feedparser.parse(content)
        articles = []
        
        # Tasks for fetching individual article content
        tasks = []
        for entry in feed.entries:
            if len(self.all_articles) + len(articles) >= self.collection_config.target_article_count:
                break
            tasks.append(self.process_rss_entry(session, entry, source))
        
        # Run article fetches concurrently (they will still respect domain locks internally)
        results = await asyncio.gather(*tasks)
        for article in results:
            if article:
                articles.append(article)
        
        logger.info(f"Collected {len(articles)} articles from {source.name}")
        return articles

    async def process_rss_entry(self, session: aiohttp.ClientSession, entry, source: SourceConfig) -> Optional[Article]:
        """Process a single RSS entry asynchronously"""
        title = entry.get('title', '').strip()
        url = entry.get('link', '')
        if not title or not url:
            return None
        
        # Check robots.txt (blocking call, but fast as it's cached)
        if self.collection_config.respect_robots_txt:
            loop = asyncio.get_running_loop()
            can_fetch = await loop.run_in_executor(None, self.robots_checker.can_fetch, url)
            if not can_fetch:
                return None

        # Check date (reuse sync logic)
        published_at = self._parse_date(entry)
        if not published_at:
            return None

        # Content - try summary first, then fetch full
        summary = entry.get('summary', '') or entry.get('description', '')
        content = ""
        
        if len(summary) >= self.collection_config.min_content_length:
            content = self._clean_html(summary)
        else:
            # Scrape full article asynchronously
            full_html = await self.fetch_url(session, url)
            if full_html:
                content = self._extract_content(full_html)
        
        if not content or len(content) < self.collection_config.min_content_length:
            return None
            
        article = Article(
            title=title,
            content=content,
            source=source.name,
            url=url,
            published_at=published_at,
            author=entry.get('author'),
            category=source.category,
            tags=[tag.term for tag in entry.get('tags', [])]
        )
        
        return article

    def _parse_date(self, entry) -> Optional[datetime]:
        """Parse date from entry (mostly sync logic)"""
        date_fields = ['published_parsed', 'updated_parsed']
        for field in date_fields:
            if field in entry and entry[field]:
                try:
                    return datetime(*entry[field][:6])
                except: continue
        return None

    def _clean_html(self, html: str) -> str:
        soup = BeautifulSoup(html, 'html.parser')
        return ' '.join(soup.get_text().split())

    def _extract_content(self, html: str) -> str:
        """Extract main content using heuristics"""
        soup = BeautifulSoup(html, 'html.parser')
        # Remove junk
        for tag in soup(['script', 'style', 'nav', 'aside', 'footer', 'header']):
            tag.decompose()
        
        # Try common selectors
        selectors = ['article', '.article-content', '.post-content', '.entry-content', 'main']
        for s in selectors:
            div = soup.select_one(s)
            if div:
                text = ' '.join(div.get_text(separator=' ').split())
                if len(text) > 200:
                    return text
        
        # Fallback: all paragraphs
        paragraphs = soup.find_all('p')
        text = ' '.join(p.get_text().split() for p in paragraphs)
        return text

    async def run(self) -> List[Article]:
        """Main entry point for async collection"""
        logger.info("Starting ASYNC News Collection Process")
        
        import socket
        connector = aiohttp.TCPConnector(family=socket.AF_INET, ssl=self.ssl_context, force_close=True)
        async with aiohttp.ClientSession(headers={'User-Agent': self.collection_config.user_agent}, connector=connector) as session:
            sources = Config.get_enabled_sources()
            tasks = []
            
            for source in sources:
                if source.source_type == 'rss':
                    tasks.append(self.collect_from_rss(session, source))
                # Add more source types here if needed
            
            results = await asyncio.gather(*tasks)
            for articles in results:
                self.all_articles.extend(articles)
        
        logger.info(f"Async collection complete: {len(self.all_articles)} articles collected")
        return self.all_articles

def run_async_collection(collection_config=None):
    """Wrapper to run the async collector from sync code"""
    collector = AsyncNewsCollector(collection_config)
    return asyncio.run(collector.run())
