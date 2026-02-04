import logging
import logging.config
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
from datetime import datetime

from .config import Config
from .robots_checker import RobotsChecker
from .source_diversity import SourceDiversityAnalyzer
from .collectors.base import Article
from .collectors.rss_collector import RSSCollector
from .collectors.api_collector import APICollector
from .collectors.web_scraper import WebScraper
from .async_collector import AsyncNewsCollector

# Logging setup
Config.create_directories()
logging.config.dictConfig(Config.LOG_CONFIG)
logger = logging.getLogger(__name__)

class NewsCollectionOrchestrator:
    """Main orchestrator for news collection process"""
    
    def __init__(self, collection_config=None):
        self.collection_config = collection_config or Config.DEFAULT_COLLECTION
        self.robots_checker = RobotsChecker(
            user_agent=self.collection_config.user_agent,
            cache_dir=Config.CACHE_DIR
        )
        self.diversity_analyzer = SourceDiversityAnalyzer(Config.get_enabled_sources())
        self.all_articles: List[Article] = []
        self.collectors = []
        
        logger.info("NewsCollectionOrchestrator initialized")
        logger.info(f"Collection period: {self.collection_config.start_date.date()} to {self.collection_config.end_date.date()}")
        logger.info(f"Target article count: {self.collection_config.target_article_count}")
        logger.info(f"Keywords: {', '.join(self.collection_config.keywords[:5])}...")
    
    def create_collectors(self) -> List:
        """Create collectors based on source configs"""
        collectors = []
        
        for source_config in Config.get_enabled_sources():
            try:
                if source_config.source_type == 'rss':
                    collector = RSSCollector(
                        source_config,
                        self.collection_config,
                        self.robots_checker
                    )
                elif source_config.source_type == 'api':
                    collector = APICollector(
                        source_config,
                        self.collection_config,
                        self.robots_checker
                    )
                elif source_config.source_type == 'scraper':
                    collector = WebScraper(
                        source_config,
                        self.collection_config,
                        self.robots_checker
                    )
                else:
                    logger.warning(f"Unknown source type: {source_config.source_type}")
                    continue
                
                collectors.append(collector)
                logger.info(f"Created {collector.__class__.__name__} for {source_config.name}")
                
            except Exception as e:
                logger.error(f"Error creating collector for {source_config.name}: {e}")
        
        return collectors
    
    def collect_sequential(self) -> List[Article]:
        """Collect data sequentially (safer)"""
        logger.info("Starting sequential collection")
        
        self.collectors = self.create_collectors()
        
        for collector in self.collectors:
            try:
                logger.info(f"Collecting from {collector.source_config.name}...")
                articles = collector.collect()
                self.all_articles.extend(articles)
                logger.info(f"Collected {len(articles)} articles from {collector.source_config.name}")
                
                # Check if target reached
                if len(self.all_articles) >= self.collection_config.target_article_count:
                    logger.info(f"Target article count reached: {len(self.all_articles)}")
                    break
                    
            except Exception as e:
                logger.error(f"Error in sequential collection from {collector.source_config.name}: {e}")
        
        return self.all_articles
    
    def collect_parallel(self) -> List[Article]:
        """Collect data in parallel (faster)"""
        logger.info("Starting parallel collection")
        
        self.collectors = self.create_collectors()
        
        with ThreadPoolExecutor(max_workers=self.collection_config.max_workers) as executor:
            # Create task for each collector
            future_to_collector = {
                executor.submit(collector.collect): collector 
                for collector in self.collectors
            }
            
            # Wait for tasks to complete
            for future in as_completed(future_to_collector):
                collector = future_to_collector[future]
                try:
                    articles = future.result()
                    self.all_articles.extend(articles)
                    logger.info(f"Collected {len(articles)} articles from {collector.source_config.name}")
                except Exception as e:
                    logger.error(f"Error in parallel collection from {collector.source_config.name}: {e}")
        
        return self.all_articles
    
    def collect_async(self) -> List[Article]:
        """Collect data asynchronously (fastest)"""
        logger.info("Starting asynchronous collection")
        
        # We'll use a wrapper to run the async loop
        import asyncio
        collector = AsyncNewsCollector(self.collection_config)
        self.all_articles = asyncio.run(collector.run())
        
        # We still want to keep the collectors list for report generation
        self.collectors = self.create_collectors()
        
        return self.all_articles

    def remove_duplicates(self):
        """Remove duplicate articles"""
        before_count = len(self.all_articles)
        
        # Deduplicate by ID
        seen_ids = set()
        unique_articles = []
        
        for article in self.all_articles:
            if article.article_id not in seen_ids:
                seen_ids.add(article.article_id)
                unique_articles.append(article)
        
        self.all_articles = unique_articles
        after_count = len(self.all_articles)
        
        removed = before_count - after_count
        if removed > 0:
            logger.info(f"Removed {removed} duplicate articles")
    
    def analyze_diversity(self):
        """Analyze diversity of collected articles"""
        logger.info("Analyzing source diversity...")
        
        articles_dict = [article.to_dict() for article in self.all_articles]
        metrics = self.diversity_analyzer.analyze_collection(articles_dict)
        
        # Generate and log report
        report = self.diversity_analyzer.generate_report(metrics)
        print("\n" + report)
        
        # Quality check
        passed, issues = self.diversity_analyzer.check_quality_thresholds(metrics)
        if not passed:
            logger.warning("Quality thresholds not met:")
            for issue in issues:
                logger.warning(f"  - {issue}")
        else:
            logger.info("All quality thresholds passed!")
        
        return metrics
    
    def save_results(self, output_format='json', filename=None):
        """Save results"""
        if not self.all_articles:
            logger.warning("No articles to save")
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if filename is None:
            filename = f"news_collection_{timestamp}"
        
        articles_dict = [article.to_dict() for article in self.all_articles]
        
        if output_format == 'json':
            filepath = Config.RAW_DATA_DIR / f"{filename}.json"
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(articles_dict, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(articles_dict)} articles to {filepath}")
        
        elif output_format == 'csv':
            import csv
            filepath = Config.RAW_DATA_DIR / f"{filename}.csv"
            
            fieldnames = ['id', 'title', 'content', 'source', 'url', 'published_at', 
                         'author', 'category', 'tags', 'collected_at']
            
            with open(filepath, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for article in articles_dict:
                    writer.writerow({
                        'id': article['id'],
                        'title': article['title'],
                        'content': article['content'][:1000],  # First 1000 chars
                        'source': article['source'],
                        'url': article['url'],
                        'published_at': article['published_at'],
                        'author': article.get('author', ''),
                        'category': article.get('category', ''),
                        'tags': ','.join(article.get('tags', [])),
                        'collected_at': article['collected_at']
                    })
            
            logger.info(f"Saved {len(articles_dict)} articles to {filepath}")
        
        return filepath
    
    def generate_collection_report(self) -> Dict:
        """Generate general report of collection process"""
        report = {
            'collection_date': datetime.now().isoformat(),
            'total_articles_collected': len(self.all_articles),
            'collection_config': {
                'start_date': self.collection_config.start_date.isoformat(),
                'end_date': self.collection_config.end_date.isoformat(),
                'target_count': self.collection_config.target_article_count,
                'keywords': self.collection_config.keywords
            },
            'sources': []
        }
        
        for collector in self.collectors:
            source_report = {
                'name': collector.source_config.name,
                'type': collector.source_config.source_type,
                'reliability': collector.source_config.reliability_score,
                'stats': collector.get_stats()
            }
            report['sources'].append(source_report)
        
        return report
    
    def run(self, parallel=True, output_format='json'):
        """Run main collection process"""
        logger.info("=" * 60)
        logger.info("STARTING NEWS COLLECTION PROCESS")
        logger.info("=" * 60)
        
        # Data collection
        if parallel == 'async':
            self.collect_async()
        elif parallel:
            self.collect_parallel()
        else:
            self.collect_sequential()
        
        logger.info(f"Collection complete: {len(self.all_articles)} articles collected")
        
        # Remove duplicates
        self.remove_duplicates()
        
        # Diversity analysis
        diversity_metrics = self.analyze_diversity()
        
        # Save results
        filepath = self.save_results(output_format=output_format)
        
        # General report
        collection_report = self.generate_collection_report()
        report_path = Config.RAW_DATA_DIR / f"collection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(collection_report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Collection report saved to {report_path}")
        logger.info("=" * 60)
        logger.info("COLLECTION PROCESS COMPLETED")
        logger.info("=" * 60)
        
        return {
            'articles': self.all_articles,
            'diversity_metrics': diversity_metrics,
            'collection_report': collection_report,
            'output_file': str(filepath)
        }

# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='News Collection System')
    parser.add_argument('--parallel', action='store_true', help='Use parallel collection')
    parser.add_argument('--format', choices=['json', 'csv'], default='json', help='Output format')
    parser.add_argument('--target', type=int, help='Target article count')
    parser.add_argument('--days', type=int, default=90, help='Number of days to look back')
    
    args = parser.parse_args()
    
    # Customize config
    if args.target:
        Config.DEFAULT_COLLECTION.target_article_count = args.target
    
    if args.days:
        from datetime import timedelta
        Config.DEFAULT_COLLECTION.start_date = datetime.now() - timedelta(days=args.days)
    
    # Run orchestrator
    orchestrator = NewsCollectionOrchestrator()
    results = orchestrator.run(parallel=args.parallel, output_format=args.format)
    
    print(f"\n✓ Collection completed successfully!")
    print(f"✓ Total articles: {len(results['articles'])}")
    print(f"✓ Output file: {results['output_file']}")