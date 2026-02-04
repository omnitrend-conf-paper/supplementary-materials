import logging
import requests
from typing import List, Dict
from datetime import datetime
import time
from .base import BaseCollector, Article

logger = logging.getLogger(__name__)

class APICollector(BaseCollector):
    """Collector for API sources"""
    
    def __init__(self, source_config, collection_config, robots_checker=None):
        super().__init__(source_config, collection_config, robots_checker)
        self.headers = {
            'User-Agent': collection_config.user_agent
        }
        self.request_count = 0
        self.last_request_time = None
    
    def collect(self) -> List[Article]:
        """Collect articles from API"""
        logger.info(f"Starting API collection from {self.source_config.name}")
        
        if not self.source_config.api_key:
            logger.error(f"No API key configured for {self.source_config.name}")
            return self.collected_articles
        
        try:
            # Special implementation for NewsAPI
            if 'newsapi' in self.source_config.url.lower():
                self._collect_from_newsapi()
            else:
                logger.warning(f"Unsupported API type: {self.source_config.url}")
            
            self.log_stats()
            
        except Exception as e:
            logger.error(f"Error collecting from API {self.source_config.url}: {e}")
        
        return self.collected_articles
    
    def _collect_from_newsapi(self):
        """Collect from NewsAPI"""
        page = 1
        total_pages = 1
        
        while page <= total_pages and self.stats['successful'] < self.collection_config.target_article_count:
            # Rate limit check
            self._check_rate_limit()
            
            params = {
                'apiKey': self.source_config.api_key,
                'q': ' OR '.join(self.collection_config.keywords[:5]),  # First 5 keywords
                'language': 'tr',
                'sortBy': 'publishedAt',
                'pageSize': 100,
                'page': page,
                'from': self.collection_config.start_date.strftime('%Y-%m-%d'),
                'to': self.collection_config.end_date.strftime('%Y-%m-%d')
            }
            
            try:
                response = requests.get(
                    self.source_config.url,
                    params=params,
                    headers=self.headers,
                    timeout=15
                )
                response.raise_for_status()
                
                data = response.json()
                
                if data.get('status') != 'ok':
                    logger.error(f"API error: {data.get('message')}")
                    break
                
                articles = data.get('articles', [])
                total_results = data.get('totalResults', 0)
                
                logger.info(f"Page {page}: Found {len(articles)} articles (Total: {total_results})")
                
                # Calculate total pages
                if page == 1 and total_results > 0:
                    total_pages = min((total_results + 99) // 100, 10)  # Max 10 pages
                
                # Process each article
                for article_data in articles:
                    try:
                        article = self._parse_newsapi_article(article_data)
                        if article:
                            self.add_article(article)
                    except Exception as e:
                        logger.error(f"Error parsing article: {e}")
                        self.stats['failed'] += 1
                        continue
                
                page += 1
                
                # Last page or target reached?
                if len(articles) == 0 or self.stats['successful'] >= self.collection_config.target_article_count:
                    break
                
            except requests.exceptions.RequestException as e:
                logger.error(f"API request error: {e}")
                self.stats['failed'] += 1
                break
    
    def _parse_newsapi_article(self, data: Dict) -> Article:
        """Parse NewsAPI article"""
        title = data.get('title', '').strip()
        if not title or title == '[Removed]':
            return None
        
        content = data.get('content', '') or data.get('description', '')
        if not content or content == '[Removed]':
            return None
        
        url = data.get('url', '')
        if not url:
            return None
        
        # Date
        published_str = data.get('publishedAt', '')
        try:
            published_at = datetime.fromisoformat(published_str.replace('Z', '+00:00'))
        except:
            logger.debug(f"Could not parse date: {published_str}")
            return None
        
        # Source name
        source_name = data.get('source', {}).get('name', self.source_config.name)
        
        # Author
        author = data.get('author')
        
        article = Article(
            title=title,
            content=content,
            source=source_name,
            url=url,
            published_at=published_at,
            author=author,
            category=self.source_config.category
        )
        
        return article
    
    def _check_rate_limit(self):
        """Check rate limit"""
        current_time = time.time()
        
        if self.last_request_time is not None:
            elapsed = current_time - self.last_request_time
            min_delay = 3600 / self.source_config.rate_limit  # seconds
            
            if elapsed < min_delay:
                sleep_time = min_delay - elapsed
                logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)
        
        self.last_request_time = time.time()
        self.request_count += 1