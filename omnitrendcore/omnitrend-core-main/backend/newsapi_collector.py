#!/usr/bin/env python3
"""
NewsAPI Collector for real-time news data
Free tier: 100 requests/day, 1000 results max per query

Usage:
    export NEWSAPI_KEY="your_api_key"
    python newsapi_collector.py --query "technology" --max-articles 100
"""

import os
import json
import argparse
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import time


class NewsAPICollector:
    """Collect news articles from NewsAPI.org"""
    
    BASE_URL = "https://newsapi.org/v2"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("NEWSAPI_KEY")
        if not self.api_key:
            raise ValueError("NewsAPI key required. Set NEWSAPI_KEY env variable or pass api_key param")
        
        self.session = requests.Session()
        self.session.headers.update({
            "X-Api-Key": self.api_key,
            "User-Agent": "OmniTrend/1.0"
        })
    
    def get_top_headlines(
        self,
        country: str = "us",
        category: Optional[str] = None,
        query: Optional[str] = None,
        page_size: int = 100
    ) -> List[Dict]:
        """Fetch top headlines"""
        params = {
            "country": country,
            "pageSize": min(page_size, 100)
        }
        if category:
            params["category"] = category
        if query:
            params["q"] = query
            
        response = self.session.get(f"{self.BASE_URL}/top-headlines", params=params)
        response.raise_for_status()
        data = response.json()
        
        if data.get("status") != "ok":
            raise Exception(f"API error: {data.get('message')}")
        
        return self._parse_articles(data.get("articles", []))
    
    def search_everything(
        self,
        query: str,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        language: str = "en",
        sort_by: str = "publishedAt",
        max_articles: int = 100
    ) -> List[Dict]:
        """Search all articles matching query"""
        if from_date is None:
            from_date = datetime.now() - timedelta(days=7)  # Free tier: last 7 days
        if to_date is None:
            to_date = datetime.now()
        
        all_articles = []
        page = 1
        
        while len(all_articles) < max_articles:
            params = {
                "q": query,
                "from": from_date.strftime("%Y-%m-%d"),
                "to": to_date.strftime("%Y-%m-%d"),
                "language": language,
                "sortBy": sort_by,
                "pageSize": min(100, max_articles - len(all_articles)),
                "page": page
            }
            
            response = self.session.get(f"{self.BASE_URL}/everything", params=params)
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") != "ok":
                raise Exception(f"API error: {data.get('message')}")
            
            articles = data.get("articles", [])
            if not articles:
                break
            
            all_articles.extend(self._parse_articles(articles))
            
            # Check if more pages available
            total_results = data.get("totalResults", 0)
            if page * 100 >= total_results or page >= 5:  # Max 5 pages (500 articles)
                break
            
            page += 1
            time.sleep(0.5)  # Rate limiting
        
        return all_articles[:max_articles]
    
    def collect_by_categories(
        self,
        categories: List[str] = None,
        country: str = "us",
        max_per_category: int = 50
    ) -> List[Dict]:
        """Collect headlines from multiple categories"""
        if categories is None:
            categories = ["business", "technology", "science", "health", "sports", "entertainment"]
        
        all_articles = []
        
        for category in categories:
            print(f"Collecting {category}...")
            try:
                articles = self.get_top_headlines(
                    country=country,
                    category=category,
                    page_size=max_per_category
                )
                for art in articles:
                    art["category"] = category
                all_articles.extend(articles)
                time.sleep(0.5)  # Rate limiting
            except Exception as e:
                print(f"  Error: {e}")
                continue
        
        return all_articles
    
    def _parse_articles(self, raw_articles: List[Dict]) -> List[Dict]:
        """Parse raw API articles to our format"""
        parsed = []
        
        for art in raw_articles:
            title = art.get("title", "").strip()
            if not title or title == "[Removed]":
                continue
            
            content = art.get("content") or art.get("description") or ""
            if content == "[Removed]":
                content = art.get("description", "")
            
            # Parse date
            pub_str = art.get("publishedAt", "")
            try:
                published_at = datetime.fromisoformat(pub_str.replace("Z", "+00:00")).isoformat()
            except:
                published_at = datetime.now().isoformat()
            
            parsed.append({
                "id": None,  # Will be assigned during processing
                "title": title,
                "content": content[:2000],  # Limit content length
                "source": art.get("source", {}).get("name", "NewsAPI"),
                "url": art.get("url", ""),
                "published_at": published_at,
                "author": art.get("author"),
                "image_url": art.get("urlToImage")
            })
        
        return parsed
    
    def save_articles(self, articles: List[Dict], output_path: str):
        """Save articles to JSON file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(articles)} articles to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Collect news from NewsAPI")
    parser.add_argument("--query", type=str, help="Search query")
    parser.add_argument("--category", type=str, help="Category for headlines")
    parser.add_argument("--country", type=str, default="us", help="Country code")
    parser.add_argument("--max-articles", type=int, default=100, help="Max articles to collect")
    parser.add_argument("--output", type=str, default="data/raw/newsapi_articles.json")
    parser.add_argument("--all-categories", action="store_true", help="Collect from all categories")
    args = parser.parse_args()
    
    try:
        collector = NewsAPICollector()
    except ValueError as e:
        print(f"Error: {e}")
        print("\nTo get a free API key:")
        print("  1. Go to https://newsapi.org/register")
        print("  2. Sign up for free account")
        print("  3. Copy your API key")
        print("  4. Run: export NEWSAPI_KEY='your_key_here'")
        return
    
    if args.all_categories:
        articles = collector.collect_by_categories(
            country=args.country,
            max_per_category=args.max_articles // 6
        )
    elif args.query:
        articles = collector.search_everything(
            query=args.query,
            max_articles=args.max_articles
        )
    else:
        articles = collector.get_top_headlines(
            country=args.country,
            category=args.category,
            page_size=args.max_articles
        )
    
    print(f"\nCollected {len(articles)} articles")
    
    if articles:
        collector.save_articles(articles, args.output)
        
        # Print sample
        print("\nSample articles:")
        for art in articles[:3]:
            print(f"  - {art['title'][:60]}...")
            print(f"    Source: {art['source']}, Date: {art['published_at'][:10]}")


if __name__ == "__main__":
    main()
