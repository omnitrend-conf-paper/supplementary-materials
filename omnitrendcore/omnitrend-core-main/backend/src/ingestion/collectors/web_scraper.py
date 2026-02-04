import logging
import requests
from bs4 import BeautifulSoup
from typing import List, Optional
from datetime import datetime
import re
from .base import BaseCollector, Article

logger = logging.getLogger(__name__)

class WebScraper(BaseCollector):
    """Web scraper collector"""
    
    def __init__(self, source_config, collection_config, robots_checker=None):
        super().__init__(source_config, collection_config, robots_checker)
        self.headers = {
            'User-Agent': collection_config.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml',
            'Accept-Language': 'tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def collect(self) -> List[Article]:
        """Scrape articles from website"""
        logger.info(f"Starting web scraping from {self.source_config.name}")
        
        if not self.can_fetch_url(self.source_config.url):
            logger.error(f"Cannot fetch {self.source_config.url} due to robots.txt")
            return self.collected_articles
        
        try:
            # Collect article URLs from homepage
            article_urls = self._discover_article_urls()
            
            logger.info(f"Found {len(article_urls)} potential article URLs")
            
            # Scrape each URL
            for url in article_urls:
                if self.stats['successful'] >= self.collection_config.target_article_count:
                    logger.info("Target article count reached")
                    break
                
                try:
                    article = self._scrape_article(url)
                    if article:
                        self.add_article(article)
                except Exception as e:
                    logger.error(f"Error scraping article from {url}: {e}")
                    self.stats['failed'] += 1
                    self.failed_urls.append(url)
                    continue
            
            self.log_stats()
            
        except Exception as e:
            logger.error(f"Error during web scraping from {self.source_config.url}: {e}")
        finally:
            self.session.close()
        
        return self.collected_articles
    
    def _discover_article_urls(self) -> List[str]:
        """Discover article URLs from homepage"""
        article_urls = set()
        
        try:
            # Rate limiting
            self.respect_rate_limit(self.source_config.url)
            
            response = self.session.get(self.source_config.url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find article links
            # Usually URLs containing 'article', 'news', 'haber', etc.
            link_patterns = [
                r'/haber/',
                r'/makale/',
                r'/article/',
                r'/news/',
                r'/\d{4}/\d{2}/',  # Date-based URLs
            ]
            
            for link in soup.find_all('a', href=True):
                href = link['href']
                
                # Convert relative URLs to absolute
                if href.startswith('/'):
                    href = f"{self.source_config.url.rstrip('/')}{href}"
                elif not href.startswith('http'):
                    continue
                
                # URL pattern check
                if any(re.search(pattern, href) for pattern in link_patterns):
                    # Same domain?
                    if self.source_config.url.split('/')[2] in href:
                        article_urls.add(href)
            
            # Also check sub-pages (pagination)
            pagination_links = soup.find_all('a', class_=re.compile(r'page|next|pagination'))
            for link in pagination_links[:3]:  # First 3 pages
                if link.get('href'):
                    page_url = link['href']
                    if page_url.startswith('/'):
                        page_url = f"{self.source_config.url.rstrip('/')}{page_url}"
                    
                    if page_url != self.source_config.url:
                        page_urls = self._discover_urls_from_page(page_url)
                        article_urls.update(page_urls)
            
        except Exception as e:
            logger.error(f"Error discovering article URLs: {e}")
        
        return list(article_urls)[:100]  # Max 100 URLs
    
    def _discover_urls_from_page(self, page_url: str) -> List[str]:
        """Discover URLs from specific page"""
        urls = set()
        
        try:
            if not self.can_fetch_url(page_url):
                return []
            
            self.respect_rate_limit(page_url)
            
            response = self.session.get(page_url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href.startswith('/'):
                    href = f"{self.source_config.url.rstrip('/')}{href}"
                
                if 'haber' in href or 'article' in href or 'news' in href:
                    urls.add(href)
        
        except Exception as e:
            logger.debug(f"Error discovering URLs from {page_url}: {e}")
        
        return list(urls)
    
    def _scrape_article(self, url: str) -> Optional[Article]:
        """Scrape single article"""
        if not self.can_fetch_url(url):
            return None
        
        # Rate limiting
        self.respect_rate_limit(url)
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Get selectors from scraper config
            config = self.source_config.scraper_config
            
            # Title
            title = self._extract_title(soup, config.get('title_selector'))
            if not title:
                return None
            
            # Content
            content = self._extract_content(soup, config.get('content_selector'))
            if not content:
                return None
            
            # Date
            published_at = self._extract_date(soup, config.get('date_selector'))
            if not published_at:
                # Try to get date from meta tags
                published_at = self._extract_date_from_meta(soup)
            
            if not published_at:
                logger.debug(f"Could not extract date from {url}")
                return None
            
            # Author
            author = self._extract_author(soup, config.get('author_selector'))
            
            article = Article(
                title=title,
                content=content,
                source=self.source_config.name,
                url=url,
                published_at=published_at,
                author=author,
                category=self.source_config.category
            )
            
            return article
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return None
    
    def _extract_title(self, soup: BeautifulSoup, selector: Optional[str]) -> str:
        """Extract title"""
        # Try given selector first
        if selector:
            element = soup.select_one(selector)
            if element:
                return element.get_text(strip=True)
        
        # Try common title selectors
        selectors = ['h1.title', 'h1.article-title', 'h1', 'title']
        for sel in selectors:
            element = soup.select_one(sel)
            if element:
                title = element.get_text(strip=True)
                if len(title) > 10:  # At least 10 characters
                    return title
        
        # Get from meta tag
        meta_title = soup.find('meta', property='og:title')
        if meta_title and meta_title.get('content'):
            return meta_title['content']
        
        return ''
    
    def _extract_content(self, soup: BeautifulSoup, selector: Optional[str]) -> str:
        """Extract content"""
        # Try given selector first
        if selector:
            element = soup.select_one(selector)
            if element:
                return self._clean_content(element)
        
        # Try common content selectors
        selectors = [
            'article',
            '.article-content',
            '.post-content',
            '.entry-content',
            '.content',
            'main article',
            '[itemprop="articleBody"]'
        ]
        
        for sel in selectors:
            element = soup.select_one(sel)
            if element:
                content = self._clean_content(element)
                if len(content) >= self.collection_config.min_content_length:
                    return content
        
        # If nothing else, get all p tags
        paragraphs = soup.find_all('p')
        content = ' '.join(p.get_text(strip=True) for p in paragraphs)
        
        return content
    
    def _extract_date(self, soup: BeautifulSoup, selector: Optional[str]) -> Optional[datetime]:
        """Extract date"""
        date_str = None
        
        # Try given selector
        if selector:
            element = soup.select_one(selector)
            if element:
                date_str = element.get_text(strip=True)
        
        # Try common date selectors
        if not date_str:
            selectors = ['.publish-date', '.date', 'time', '[itemprop="datePublished"]']
            for sel in selectors:
                element = soup.select_one(sel)
                if element:
                    date_str = element.get('datetime') or element.get_text(strip=True)
                    if date_str:
                        break
        
        if date_str:
            return self._parse_date_string(date_str)
        
        return None
    
    def _extract_date_from_meta(self, soup: BeautifulSoup) -> Optional[datetime]:
        """Extract date from meta tags"""
        meta_tags = [
            ('property', 'article:published_time'),
            ('name', 'publishdate'),
            ('name', 'publish_date'),
            ('property', 'og:published_time')
        ]
        
        for attr, value in meta_tags:
            meta = soup.find('meta', {attr: value})
            if meta and meta.get('content'):
                return self._parse_date_string(meta['content'])
        
        return None
    
    def _parse_date_string(self, date_str: str) -> Optional[datetime]:
        """Parse date string"""
        # ISO format
        try:
            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        except:
            pass
        
        # Other common formats
        formats = [
            '%Y-%m-%d',
            '%d.%m.%Y',
            '%d/%m/%Y',
            '%Y-%m-%d %H:%M:%S',
            '%d.%m.%Y %H:%M'
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except:
                continue
        
        return None
    
    def _extract_author(self, soup: BeautifulSoup, selector: Optional[str]) -> Optional[str]:
        """Extract author info"""
        if selector:
            element = soup.select_one(selector)
            if element:
                return element.get_text(strip=True)
        
        # Get from meta tag
        meta_author = soup.find('meta', attrs={'name': 'author'})
        if meta_author and meta_author.get('content'):
            return meta_author['content']
        
        return None
    
    def _clean_content(self, element) -> str:
        """Clean content"""
        # Remove unwanted tags
        for tag in element(['script', 'style', 'nav', 'aside', 'footer', 'header', 'form']):
            tag.decompose()
        
        # Get text
        text = element.get_text(separator=' ', strip=True)
        
        # Clean multiple spaces
        text = ' '.join(text.split())
        
        return text