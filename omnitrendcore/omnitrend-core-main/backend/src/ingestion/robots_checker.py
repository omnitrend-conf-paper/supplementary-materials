import urllib.robotparser
from urllib.parse import urlparse, urljoin
import urllib.request
import ssl
import logging
from typing import Dict, Optional
import time
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)

class RobotsChecker:
    """Robots.txt checker and manager"""
    
    def __init__(self, user_agent: str, cache_dir: Path):
        self.user_agent = user_agent
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = cache_dir / "robots_cache.pkl"
        self.parsers: Dict[str, urllib.robotparser.RobotFileParser] = {}
        self.last_check: Dict[str, float] = {}
        self.cache_duration = 86400  # 24 hours
        self._load_cache()
    
    def _load_cache(self):
        """Load robots.txt info from cache"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.parsers = cache_data.get('parsers', {})
                    self.last_check = cache_data.get('last_check', {})
                logger.info(f"Robots cache loaded: {len(self.parsers)} domains")
            except Exception as e:
                logger.warning(f"Could not load robots cache: {e}")
    
    def _save_cache(self):
        """Save robots.txt info to cache"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump({
                    'parsers': self.parsers,
                    'last_check': self.last_check
                }, f)
            logger.debug("Robots cache saved")
        except Exception as e:
            logger.warning(f"Could not save robots cache: {e}")
    
    def _get_robots_url(self, url: str) -> str:
        """Build robots.txt URL from base URL"""
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    
    def _get_domain(self, url: str) -> str:
        """Extract domain from URL"""
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}"
    
    def _should_refresh(self, domain: str) -> bool:
        """Check if cache should be refreshed"""
        if domain not in self.last_check:
            return True
        return time.time() - self.last_check[domain] > self.cache_duration
    
    def fetch_robots_txt(self, url: str) -> Optional[urllib.robotparser.RobotFileParser]:
        """Fetch and parse robots.txt"""
        domain = self._get_domain(url)
        
        # Check cache
        if domain in self.parsers and not self._should_refresh(domain):
            logger.debug(f"Using cached robots.txt for {domain}")
            return self.parsers[domain]
        
        robots_url = self._get_robots_url(url)
        parser = urllib.robotparser.RobotFileParser()
        parser.set_url(robots_url)
        
        try:
            # Bypass SSL verification for robots.txt fetching
            if hasattr(ssl, '_create_unverified_context'):
                ssl_context = ssl._create_unverified_context()
                with urllib.request.urlopen(robots_url, context=ssl_context, timeout=10) as response:
                    content = response.read().decode('utf-8')
                    parser.parse(content.splitlines())
            else:
                parser.read()
                
            self.parsers[domain] = parser
            self.last_check[domain] = time.time()
            self._save_cache()
            logger.info(f"Fetched robots.txt from {robots_url}")
            return parser
        except Exception as e:
            logger.warning(f"Could not fetch robots.txt from {robots_url}: {e}")
            # If no robots.txt or inaccessible, allow by default
            return None
    
    def can_fetch(self, url: str) -> bool:
        """Check if URL can be crawled"""
        parser = self.fetch_robots_txt(url)
        
        if parser is None:
            # If no robots.txt or inaccessible, allow
            return True
        
        try:
            can_fetch = parser.can_fetch(self.user_agent, url)
            logger.debug(f"Can fetch {url}: {can_fetch}")
            return can_fetch
        except Exception as e:
            logger.error(f"Error checking robots.txt for {url}: {e}")
            return False
    
    def get_crawl_delay(self, url: str) -> Optional[float]:
        """Get crawl delay value"""
        parser = self.fetch_robots_txt(url)
        
        if parser is None:
            return None
        
        try:
            delay = parser.crawl_delay(self.user_agent)
            if delay is not None:
                logger.info(f"Crawl delay for {url}: {delay} seconds")
            return delay
        except Exception as e:
            logger.error(f"Error getting crawl delay for {url}: {e}")
            return None
    
    def clear_cache(self):
        """Clear cache"""
        self.parsers.clear()
        self.last_check.clear()
        if self.cache_file.exists():
            self.cache_file.unlink()
        logger.info("Robots cache cleared")
    
    def get_cache_info(self) -> Dict:
        """Return cache information"""
        return {
            "cached_domains": len(self.parsers),
            "domains": list(self.parsers.keys()),
            "last_check": {
                domain: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts))
                for domain, ts in self.last_check.items()
            }
        }