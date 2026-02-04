#!/usr/bin/env python3
"""
Google Trends Collector using pytrends library
No API key required - uses unofficial Google Trends API

Usage:
    python google_trends_collector.py --keywords "AI,bitcoin,climate" --timeframe "now 7-d"
"""

import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import time

try:
    from pytrends.request import TrendReq
    PYTRENDS_AVAILABLE = True
except ImportError:
    PYTRENDS_AVAILABLE = False
    print("Warning: pytrends not installed. Run: pip install pytrends")


class GoogleTrendsCollector:
    """Collect trending topics and interest data from Google Trends"""
    
    def __init__(self, language: str = "en-US", timezone: int = 360):
        if not PYTRENDS_AVAILABLE:
            raise ImportError("pytrends required. Install with: pip install pytrends")
        
        self.pytrends = TrendReq(hl=language, tz=timezone)
        self.collected_data = {}
    
    def get_trending_searches(self, country: str = "united_states") -> List[Dict]:
        """Get real-time trending searches"""
        try:
            trending = self.pytrends.trending_searches(pn=country)
            
            results = []
            for idx, topic in enumerate(trending[0].tolist()):
                results.append({
                    "topic": topic,
                    "rank": idx + 1,
                    "country": country,
                    "timestamp": datetime.now().isoformat(),
                    "type": "trending_search"
                })
            
            return results
        except Exception as e:
            print(f"Error fetching trending searches: {e}")
            return []
    
    def get_realtime_trends(self, category: str = "all", country: str = "US") -> List[Dict]:
        """Get real-time trending stories"""
        try:
            # Category codes: all, e (entertainment), b (business), t (technology), etc.
            cat_map = {
                "all": "all",
                "business": "b",
                "entertainment": "e", 
                "health": "m",
                "science": "s",
                "sports": "p",
                "technology": "t",
                "top": "h"
            }
            cat_code = cat_map.get(category.lower(), "all")
            
            trends = self.pytrends.realtime_trending_searches(pn=country)
            
            results = []
            for _, row in trends.iterrows():
                results.append({
                    "title": row.get("title", ""),
                    "entity_names": row.get("entityNames", []),
                    "articles": row.get("articles", []),
                    "timestamp": datetime.now().isoformat(),
                    "type": "realtime_trend"
                })
            
            return results[:50]  # Limit to 50
        except Exception as e:
            print(f"Error fetching realtime trends: {e}")
            return []
    
    def get_interest_over_time(
        self,
        keywords: List[str],
        timeframe: str = "now 7-d",
        geo: str = ""
    ) -> Dict:
        """Get search interest over time for keywords
        
        Timeframes:
            - 'now 1-H': Past hour
            - 'now 4-H': Past 4 hours
            - 'now 1-d': Past day
            - 'now 7-d': Past 7 days
            - 'today 1-m': Past 30 days
            - 'today 3-m': Past 90 days
            - 'today 12-m': Past 12 months
        """
        # Limit to 5 keywords per request (Google Trends limit)
        keywords = keywords[:5]
        
        try:
            self.pytrends.build_payload(keywords, timeframe=timeframe, geo=geo)
            interest_df = self.pytrends.interest_over_time()
            
            if interest_df.empty:
                return {"keywords": keywords, "data": [], "timeframe": timeframe}
            
            # Convert to list of dicts
            data = []
            for idx, row in interest_df.iterrows():
                point = {"timestamp": idx.isoformat()}
                for kw in keywords:
                    if kw in row:
                        point[kw] = int(row[kw])
                data.append(point)
            
            return {
                "keywords": keywords,
                "data": data,
                "timeframe": timeframe,
                "geo": geo,
                "collected_at": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Error fetching interest data: {e}")
            return {"keywords": keywords, "data": [], "error": str(e)}
    
    def get_related_queries(self, keyword: str, timeframe: str = "today 3-m") -> Dict:
        """Get related queries for a keyword"""
        try:
            self.pytrends.build_payload([keyword], timeframe=timeframe)
            related = self.pytrends.related_queries()
            
            result = {
                "keyword": keyword,
                "rising": [],
                "top": []
            }
            
            if keyword in related:
                # Rising queries
                rising_df = related[keyword].get("rising")
                if rising_df is not None and not rising_df.empty:
                    result["rising"] = rising_df.to_dict(orient="records")
                
                # Top queries
                top_df = related[keyword].get("top")
                if top_df is not None and not top_df.empty:
                    result["top"] = top_df.to_dict(orient="records")
            
            return result
        except Exception as e:
            print(f"Error fetching related queries: {e}")
            return {"keyword": keyword, "rising": [], "top": [], "error": str(e)}
    
    def get_related_topics(self, keyword: str, timeframe: str = "today 3-m") -> Dict:
        """Get related topics for a keyword"""
        try:
            self.pytrends.build_payload([keyword], timeframe=timeframe)
            related = self.pytrends.related_topics()
            
            result = {
                "keyword": keyword,
                "rising": [],
                "top": []
            }
            
            if keyword in related:
                rising_df = related[keyword].get("rising")
                if rising_df is not None and not rising_df.empty:
                    result["rising"] = rising_df[["topic_title", "value"]].to_dict(orient="records")
                
                top_df = related[keyword].get("top")
                if top_df is not None and not top_df.empty:
                    result["top"] = top_df[["topic_title", "value"]].to_dict(orient="records")
            
            return result
        except Exception as e:
            print(f"Error fetching related topics: {e}")
            return {"keyword": keyword, "rising": [], "top": [], "error": str(e)}
    
    def collect_comprehensive(
        self,
        keywords: List[str],
        country: str = "united_states",
        timeframe: str = "now 7-d"
    ) -> Dict:
        """Collect comprehensive trends data"""
        print(f"Collecting Google Trends data for: {keywords}")
        
        results = {
            "collected_at": datetime.now().isoformat(),
            "keywords": keywords,
            "country": country,
            "timeframe": timeframe,
            "trending_searches": [],
            "interest_over_time": {},
            "related_queries": {},
            "related_topics": {}
        }
        
        # Get trending searches
        print("  Fetching trending searches...")
        results["trending_searches"] = self.get_trending_searches(country)
        time.sleep(1)
        
        # Get interest over time
        print("  Fetching interest over time...")
        results["interest_over_time"] = self.get_interest_over_time(keywords, timeframe)
        time.sleep(1)
        
        # Get related queries and topics for each keyword
        for kw in keywords[:3]:  # Limit to avoid rate limiting
            print(f"  Fetching related data for '{kw}'...")
            results["related_queries"][kw] = self.get_related_queries(kw, timeframe)
            time.sleep(1)
            results["related_topics"][kw] = self.get_related_topics(kw, timeframe)
            time.sleep(1)
        
        return results
    
    def save_results(self, data: Dict, output_path: str):
        """Save results to JSON file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"Saved trends data to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Collect Google Trends data")
    parser.add_argument("--keywords", type=str, help="Comma-separated keywords")
    parser.add_argument("--country", type=str, default="united_states", help="Country for trends")
    parser.add_argument("--timeframe", type=str, default="now 7-d", 
                       help="Timeframe (now 1-d, now 7-d, today 1-m, etc.)")
    parser.add_argument("--output", type=str, default="data/raw/google_trends.json")
    parser.add_argument("--trending-only", action="store_true", help="Only get trending searches")
    args = parser.parse_args()
    
    if not PYTRENDS_AVAILABLE:
        print("Error: pytrends not installed")
        print("Install with: pip install pytrends")
        return
    
    collector = GoogleTrendsCollector()
    
    if args.trending_only:
        trends = collector.get_trending_searches(args.country)
        print(f"\nTop Trending Searches in {args.country}:")
        for t in trends[:20]:
            print(f"  {t['rank']}. {t['topic']}")
        
        collector.save_results({"trending_searches": trends}, args.output)
    else:
        keywords = args.keywords.split(",") if args.keywords else ["AI", "bitcoin", "climate change"]
        keywords = [k.strip() for k in keywords]
        
        results = collector.collect_comprehensive(
            keywords=keywords,
            country=args.country,
            timeframe=args.timeframe
        )
        
        print(f"\nCollected data for {len(keywords)} keywords")
        print(f"Trending searches: {len(results['trending_searches'])}")
        
        collector.save_results(results, args.output)


if __name__ == "__main__":
    main()
