import os
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass, field
from datetime import datetime, timedelta

@dataclass
class SourceConfig:
    """News source configuration"""
    name: str
    url: str
    source_type: str  # 'api', 'rss', 'scraper'
    reliability_score: float  # 0-1 reliability score
    category: str  # 'finance', 'economy', 'general'
    enabled: bool = True
    api_key: str = None
    rate_limit: int = 100  # requests per hour
    scraper_config: Dict = field(default_factory=dict)
    
@dataclass
class CollectionConfig:
    """Data collection configuration"""
    start_date: datetime
    end_date: datetime
    target_article_count: int
    keywords: List[str]
    categories: List[str]
    min_content_length: int = 200
    max_workers: int = 5
    respect_robots_txt: bool = True
    user_agent: str = "NewsCollectorBot/1.0"
    request_delay: float = 1.0  # seconds between requests

class Config:
    """Main configuration class"""
    
    # Project directory structure
    BASE_DIR = Path(__file__).parent.parent.parent  # backend/
    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    LOGS_DIR = BASE_DIR / "logs"
    CACHE_DIR = BASE_DIR / "cache"
    
    # Premium Tier-1 Financial News Sources
    SOURCES = [
        # Top-tier financial news - Highest reliability
        SourceConfig(
            name="Financial Times",
            url="https://www.ft.com/rss/home",
            source_type="rss",
            reliability_score=0.98,
            category="finance",
            rate_limit=60
        ),
        SourceConfig(
            name="Bloomberg",
            url="https://www.bloomberg.com/feed/podcast/etf-report.xml",  # Alternative: Use API or scraper
            source_type="rss",
            reliability_score=0.97,
            category="finance",
            rate_limit=60
        ),
        SourceConfig(
            name="Wall Street Journal",
            url="https://feeds.content.dowjones.io/public/rss/RSSMarketsMain",
            source_type="rss",
            reliability_score=0.97,
            category="finance",
            rate_limit=60
        ),
        SourceConfig(
            name="Reuters Business",
            url="https://www.reutersagency.com/feed/?taxonomy=best-topics&post_type=best",
            source_type="rss",
            reliability_score=0.96,
            category="finance",
            rate_limit=100
        ),
        
        # Secondary tier - High quality, broad coverage
        SourceConfig(
            name="CNBC",
            url="https://www.cnbc.com/id/100003114/device/rss/rss.html",
            source_type="rss",
            reliability_score=0.93,
            category="finance",
            rate_limit=100
        ),
        SourceConfig(
            name="MarketWatch",
            url="https://feeds.content.dowjones.io/public/rss/mw_topstories",
            source_type="rss",
            reliability_score=0.92,
            category="finance",
            rate_limit=100
        ),
        SourceConfig(
            name="Yahoo Finance",
            url="https://finance.yahoo.com/news/rssindex",
            source_type="rss",
            reliability_score=0.88,
            category="finance",
            rate_limit=150
        ),
        
        # Specialized and analytical sources
        SourceConfig(
            name="Seeking Alpha",
            url="https://seekingalpha.com/feed.xml",
            source_type="rss",
            reliability_score=0.85,
            category="finance",
            rate_limit=100
        ),
        SourceConfig(
            name="Benzinga",
            url="https://www.benzinga.com/feed",
            source_type="rss",
            reliability_score=0.84,
            category="finance",
            rate_limit=100
        ),
        
        # Central Bank & Economic Data - Critical for macro analysis
        SourceConfig(
            name="Federal Reserve",
            url="https://www.federalreserve.gov/feeds/press_all.xml",
            source_type="rss",
            reliability_score=0.99,
            category="economy",
            rate_limit=50
        ),
        SourceConfig(
            name="ECB Press",
            url="https://www.ecb.europa.eu/rss/press.html",
            source_type="rss",
            reliability_score=0.99,
            category="economy",
            rate_limit=50
        ),
        
        # API Sources (require keys)
        SourceConfig(
            name="NewsAPI",
            url="https://newsapi.org/v2/everything",
            source_type="api",
            reliability_score=0.88,
            category="general",
            api_key=os.getenv("NEWSAPI_KEY"),
            rate_limit=100
        ),
    ]
    
    # Default collection settings
    DEFAULT_COLLECTION = CollectionConfig(
        start_date=datetime.now() - timedelta(days=90),  # Last 3 months
        end_date=datetime.now(),
        target_article_count=5000,
        keywords=[
            # English keywords
            "economy", "finance", "forex", "stock market", "interest rates", 
            "inflation", "federal reserve", "central bank", "dollar", "euro", 
            "gold", "bonds", "equity", "trading", "investment", "GDP", 
            "economic growth", "recession", "monetary policy", "fiscal policy"
        ],
        categories=["finance", "economy"],
        min_content_length=200,
        max_workers=5,
        respect_robots_txt=True,
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        request_delay=1.5  # Increased delay for premium sources
    )
    
    # Database settings
    DB_CONFIG = {
        "type": "sqlite",  # or "postgresql"
        "path": DATA_DIR / "news_database.db",
        # For PostgreSQL:
        # "host": "localhost",
        # "port": 5432,
        # "database": "news_db",
        # "user": "user",
        # "password": "password"
    }
    
    # Log settings
    LOG_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "detailed"
            },
            "file": {
                "class": "logging.FileHandler",
                "filename": str(LOGS_DIR / "collector.log"),
                "level": "DEBUG",
                "formatter": "detailed"
            }
        },
        "root": {
            "level": "DEBUG",
            "handlers": ["console", "file"]
        }
    }
    
    @classmethod
    def create_directories(cls):
        """Create required directories"""
        for directory in [cls.DATA_DIR, cls.RAW_DATA_DIR, cls.PROCESSED_DATA_DIR, 
                         cls.LOGS_DIR, cls.CACHE_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_enabled_sources(cls) -> List[SourceConfig]:
        """Get enabled sources"""
        return [source for source in cls.SOURCES if source.enabled]
    
    @classmethod
    def get_sources_by_type(cls, source_type: str) -> List[SourceConfig]:
        """Filter sources by type"""
        return [source for source in cls.get_enabled_sources() 
                if source.source_type == source_type]
    
    @classmethod
    def get_sources_by_category(cls, category: str) -> List[SourceConfig]:
        """Filter sources by category"""
        return [source for source in cls.get_enabled_sources() 
                if source.category == category]
    
    @classmethod
    def get_tier1_sources(cls) -> List[SourceConfig]:
        """Get premium tier-1 sources (reliability >= 0.95)"""
        return [source for source in cls.get_enabled_sources() 
                if source.reliability_score >= 0.95]