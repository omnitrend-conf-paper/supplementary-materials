from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import logging
from datetime import datetime
import time
import hashlib

logger = logging.getLogger(__name__)

class Article:
    """News article data model"""
    
    def __init__(self, 
                 title: str,
                 content: str,
                 source: str,
                 url: str,
                 published_at: datetime,
                 author: Optional[str] = None,
                 category: Optional[str] = None,
                 tags: Optional[List[str]] = None):
        self.title = title
        self.content = content
        self.source = source
        self.url = url
        self.published_at = published_at
        self.author = author
        self.category = category
        self.tags = tags or []
        self.collected_at = datetime.now()
        self.article_id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate unique ID for article"""
        unique_string = f"{self.url}_{self.title}_{self.published_at}"
        return hashlib.md5(unique_string.encode()).hexdigest()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format"""
        return {
            'id': self.article_id,
            'title': self.title,
            'content': self.content,
            'source': self.source,
            'url': self.url,
            'published_at': self.published_at.isoformat() if isinstance(self.published_at, datetime) else self.published_at,
            'author': self.author,
            'category': self.category,
            'tags': self.tags,
            'collected_at': self.collected_at.isoformat()
        }
    
    def is_valid(self, min_content_length: int = 200) -> bool:
        """Check if article is valid"""
        if not self.title or not self.content:
            return False
        if len(self.content) < min_content_length:
            return False
        if not self.url or not self.source:
            return False
        return True

class BaseCollector(ABC):
    """Base class for all collectors"""
    
    def __init__(self, source_config, collection_config, robots_checker=None):
        self.source_config = source_config
        self.collection_config = collection_config
        self.robots_checker = robots_checker
        self.collected_articles: List[Article] = []
        self.failed_urls: List[str] = []
        self.stats = {
            'total_attempted': 0,
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'duplicate': 0
        }
        logger.info(f"Initialized {self.__class__.__name__} for {source_config.name}")
    
    @abstractmethod
    def collect(self) -> List[Article]:
        """Collect articles - must be implemented by subclasses"""
        pass
    
    def can_fetch_url(self, url: str) -> bool:
        """Check if URL can be fetched"""
        if not self.collection_config.respect_robots_txt:
            return True
        
        if self.robots_checker is None:
            logger.warning("Robots checker not configured, allowing all URLs")
            return True
        
        try:
            can_fetch = self.robots_checker.can_fetch(url)
            if not can_fetch:
                logger.info(f"URL blocked by robots.txt: {url}")
                self.stats['skipped'] += 1
            return can_fetch
        except Exception as e:
            logger.error(f"Error checking robots.txt for {url}: {e}")
            return False
    
    def get_crawl_delay(self, url: str) -> float:
        """Get crawl delay value"""
        if self.robots_checker is None:
            return self.collection_config.request_delay
        
        try:
            delay = self.robots_checker.get_crawl_delay(url)
            if delay is not None:
                return max(delay, self.collection_config.request_delay)
            return self.collection_config.request_delay
        except Exception as e:
            logger.error(f"Error getting crawl delay for {url}: {e}")
            return self.collection_config.request_delay
    
    def respect_rate_limit(self, url: str):
        """Respect rate limit"""
        delay = self.get_crawl_delay(url)
        if delay > 0:
            logger.debug(f"Waiting {delay}s before next request")
            time.sleep(delay)
    
    def is_within_date_range(self, article_date: datetime) -> bool:
        """Check if article is within date range"""
        if not isinstance(article_date, datetime):
            try:
                article_date = datetime.fromisoformat(str(article_date).replace('Z', '+00:00'))
            except:
                logger.warning(f"Could not parse date: {article_date}")
                return False
        
        return (self.collection_config.start_date <= article_date <= 
                self.collection_config.end_date)
    
    def matches_keywords(self, text: str) -> bool:
        """Check if text contains keywords"""
        if not self.collection_config.keywords:
            return True
        
        text_lower = text.lower()
        return any(keyword.lower() in text_lower 
                  for keyword in self.collection_config.keywords)
    
    def add_article(self, article: Article) -> bool:
        """Add article with validation"""
        self.stats['total_attempted'] += 1
        
        # Validation
        if not article.is_valid(self.collection_config.min_content_length):
            logger.debug(f"Invalid article skipped: {article.title}")
            self.stats['skipped'] += 1
            return False
        
        # Date check
        if not self.is_within_date_range(article.published_at):
            logger.debug(f"Article outside date range: {article.title}")
            self.stats['skipped'] += 1
            return False
        
        # Keyword check
        combined_text = f"{article.title} {article.content}"
        if not self.matches_keywords(combined_text):
            logger.debug(f"Article doesn't match keywords: {article.title}")
            self.stats['skipped'] += 1
            return False
        
        # Duplicate check
        if any(a.article_id == article.article_id for a in self.collected_articles):
            logger.debug(f"Duplicate article skipped: {article.title}")
            self.stats['duplicate'] += 1
            return False
        
        self.collected_articles.append(article)
        self.stats['successful'] += 1
        logger.info(f"Article collected: {article.title[:50]}... from {article.source}")
        return True
    
    def get_stats(self) -> Dict:
        """Return statistics"""
        return {
            **self.stats,
            'success_rate': (self.stats['successful'] / self.stats['total_attempted'] * 100 
                           if self.stats['total_attempted'] > 0 else 0)
        }
    
    def log_stats(self):
        """Log statistics"""
        stats = self.get_stats()
        logger.info(f"Collection stats for {self.source_config.name}: "
                   f"Attempted={stats['total_attempted']}, "
                   f"Successful={stats['successful']}, "
                   f"Failed={stats['failed']}, "
                   f"Skipped={stats['skipped']}, "
                   f"Duplicate={stats['duplicate']}, "
                   f"Success Rate={stats['success_rate']:.1f}%")