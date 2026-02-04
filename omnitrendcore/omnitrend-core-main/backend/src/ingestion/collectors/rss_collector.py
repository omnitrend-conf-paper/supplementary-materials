import feedparser
import logging
from typing import List
from datetime import datetime
import requests
import argparse
import ssl
from bs4 import BeautifulSoup
from .base import BaseCollector, Article

logger = logging.getLogger(__name__)

class RSSCollector(BaseCollector):
    """Collector for RSS feeds"""
    
    def __init__(self, source_config, collection_config, robots_checker=None):
        super().__init__(source_config, collection_config, robots_checker)
        self.headers = {
            'User-Agent': collection_config.user_agent
        }
    
    def collect(self) -> List[Article]:
        """Collect articles from RSS feed"""
        logger.info(f"Starting RSS collection from {self.source_config.name}")
        
        try:
            # Parse RSS feed
            if hasattr(ssl, '_create_unverified_context'):
                ssl._create_default_https_context = ssl._create_unverified_context

            feed = feedparser.parse(self.source_config.url)
            
            if feed.bozo:
                logger.error(f"Error parsing RSS feed: {feed.bozo_exception}")
                return self.collected_articles
            
            logger.info(f"Found {len(feed.entries)} entries in RSS feed")
            
            # Process each entry
            for entry in feed.entries:
                if self.stats['successful'] >= self.collection_config.target_article_count:
                    logger.info("Target article count reached")
                    break
                
                try:
                    article = self._parse_entry(entry)
                    if article:
                        self.add_article(article)
                except Exception as e:
                    logger.error(f"Error parsing RSS entry: {e}")
                    self.stats['failed'] += 1
                    continue
            
            self.log_stats()
            
        except Exception as e:
            logger.error(f"Error collecting from RSS feed {self.source_config.url}: {e}")
        
        return self.collected_articles
    
    def _parse_entry(self, entry) -> Article:
        """Convert RSS entry to Article object"""
        # Title
        title = entry.get('title', '').strip()
        if not title:
            return None
        
        # URL
        url = entry.get('link', '')
        if not url:
            return None
        
        # Robots.txt check
        if not self.can_fetch_url(url):
            return None
        
        # Date
        published_at = self._parse_date(entry)
        if not published_at:
            logger.debug(f"Could not parse date for: {title}")
            return None
        
        # Content - from RSS or scrape
        content = self._get_content(entry, url)
        if not content:
            logger.debug(f"No content found for: {title}")
            return None
        
        # Author
        author = entry.get('author', None)
        
        # Tags
        tags = [tag.term for tag in entry.get('tags', [])]
        
        article = Article(
            title=title,
            content=content,
            source=self.source_config.name,
            url=url,
            published_at=published_at,
            author=author,
            category=self.source_config.category,
            tags=tags
        )
        
        return article
    
    def _parse_date(self, entry) -> datetime:
        """Parse date from entry"""
        # Try various date fields
        date_fields = ['published_parsed', 'updated_parsed', 'created_parsed']
        
        for field in date_fields:
            if field in entry:
                try:
                    time_struct = entry[field]
                    return datetime(*time_struct[:6])
                except:
                    continue
        
        # Try string formats
        date_string_fields = ['published', 'updated', 'created']
        for field in date_string_fields:
            if field in entry:
                try:
                    return datetime.fromisoformat(entry[field].replace('Z', '+00:00'))
                except:
                    continue
        
        return None
    
    def _get_content(self, entry, url: str) -> str:
        """Get content - try RSS first, then scrape"""
        # Try content from RSS
        content = entry.get('summary', '') or entry.get('description', '')
        
        # If content is long enough, use it directly
        if len(content) >= self.collection_config.min_content_length:
            return self._clean_html(content)
        
        # If content is insufficient, scrape full article
        logger.debug(f"RSS content too short, fetching full article from {url}")
        full_content = self._scrape_full_article(url)
        
        if full_content:
            return full_content
        
        # If nothing else, return existing content
        return self._clean_html(content) if content else ''
    
    def _scrape_full_article(self, url: str) -> str:
        """Scrape full article content"""
        try:
            # Rate limiting
            self.respect_rate_limit(url)
            
            # Using verify=False because many news sites have certification chain issues
            response = requests.get(url, headers=self.headers, timeout=10, verify=False)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Try common content selectors
            content_selectors = [
                'article',
                '.article-content',
                '.post-content',
                '.entry-content',
                '.content',
                'main'
            ]
            
            for selector in content_selectors:
                content_div = soup.select_one(selector)
                if content_div:
                    # Remove script and style tags
                    for tag in content_div(['script', 'style', 'nav', 'aside', 'footer']):
                        tag.decompose()
                    
                    text = content_div.get_text(separator=' ', strip=True)
                    if len(text) >= self.collection_config.min_content_length:
                        return text
            
            # If no selector found, get all p tags from body
            paragraphs = soup.find_all('p')
            text = ' '.join(p.get_text(strip=True) for p in paragraphs)
            
            return text if len(text) >= self.collection_config.min_content_length else ''
            
        except Exception as e:
            logger.error(f"Error scraping article from {url}: {e}")
            self.failed_urls.append(url)
            return ''
    
    def _clean_html(self, html_content: str) -> str:
        """Clean HTML content"""
        if not html_content:
            return ''
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove unwanted tags
        for tag in soup(['script', 'style', 'img', 'iframe']):
            tag.decompose()
        
        # Get only text
        text = soup.get_text(separator=' ', strip=True)
        
        # Clean multiple spaces
        text = ' '.join(text.split())
        
        return text