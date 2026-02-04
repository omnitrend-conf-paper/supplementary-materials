"""
Asset Tracker - Maps news articles to financial assets

Identifies mentions of companies, stocks, and cryptocurrencies in articles
and links them to their ticker symbols for sentiment-price correlation.
"""

import re
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict


# Ticker to keywords mapping for asset detection
TICKER_MAP = {
    # Big Tech
    "AAPL": ["apple", "iphone", "ipad", "macbook", "tim cook", "ios", "app store"],
    "MSFT": ["microsoft", "windows", "azure", "xbox", "satya nadella", "bing", "office 365"],
    "GOOGL": ["google", "alphabet", "youtube", "android", "sundar pichai", "chrome", "gmail"],
    "AMZN": ["amazon", "aws", "prime", "jeff bezos", "andy jassy", "alexa", "kindle"],
    "META": ["meta", "facebook", "instagram", "whatsapp", "mark zuckerberg", "oculus"],
    "NVDA": ["nvidia", "geforce", "cuda", "jensen huang", "rtx", "gpu"],
    "TSLA": ["tesla", "elon musk", "cybertruck", "model 3", "model s", "model x", "model y", "supercharger"],
    
    # Finance
    "JPM": ["jpmorgan", "jp morgan", "chase", "jamie dimon"],
    "BAC": ["bank of america", "bofa"],
    "GS": ["goldman sachs", "goldman"],
    "V": ["visa"],
    "MA": ["mastercard"],
    
    # Retail/Consumer
    "WMT": ["walmart"],
    "COST": ["costco"],
    "NKE": ["nike"],
    "SBUX": ["starbucks"],
    "MCD": ["mcdonalds", "mcdonald's"],
    
    # Energy
    "XOM": ["exxon", "exxonmobil"],
    "CVX": ["chevron"],
    
    # Healthcare
    "JNJ": ["johnson & johnson", "j&j"],
    "PFE": ["pfizer"],
    "UNH": ["unitedhealth"],
    "MRNA": ["moderna"],
    
    # Other Tech
    "NFLX": ["netflix"],
    "CRM": ["salesforce"],
    "ORCL": ["oracle"],
    "INTC": ["intel"],
    "AMD": ["amd", "advanced micro devices", "ryzen", "radeon"],
    "IBM": ["ibm"],
    
    # Crypto (mapped to common formats)
    "BTC-USD": ["bitcoin", "btc"],
    "ETH-USD": ["ethereum", "eth", "ether"],
    "SOL-USD": ["solana", "sol"],
    "XRP-USD": ["ripple", "xrp"],
    "ADA-USD": ["cardano", "ada"],
    "DOGE-USD": ["dogecoin", "doge"],
    "BNB-USD": ["binance coin", "bnb"],
    "DOT-USD": ["polkadot", "dot"],
    "AVAX-USD": ["avalanche", "avax"],
    "MATIC-USD": ["polygon", "matic"],
}

# Reverse mapping for quick lookup
KEYWORD_TO_TICKER = {}
for ticker, keywords in TICKER_MAP.items():
    for keyword in keywords:
        if keyword not in KEYWORD_TO_TICKER:
            KEYWORD_TO_TICKER[keyword] = []
        KEYWORD_TO_TICKER[keyword].append(ticker)


class AssetTracker:
    """Tracks and links financial assets in news articles"""
    
    def __init__(self, custom_mappings: Dict[str, List[str]] = None):
        """
        Initialize asset tracker
        
        Args:
            custom_mappings: Additional ticker-to-keywords mappings
        """
        self.ticker_map = TICKER_MAP.copy()
        if custom_mappings:
            self.ticker_map.update(custom_mappings)
        
        # Build reverse mapping
        self.keyword_to_ticker = defaultdict(list)
        for ticker, keywords in self.ticker_map.items():
            for keyword in keywords:
                self.keyword_to_ticker[keyword.lower()].append(ticker)
    
    def find_assets_in_text(
        self,
        text: str,
        min_confidence: float = 0.5
    ) -> List[Dict]:
        """
        Find asset mentions in text
        
        Args:
            text: Text to search
            min_confidence: Minimum confidence threshold (0-1)
        
        Returns:
            List of detected assets with confidence scores
        """
        text_lower = text.lower()
        detected = {}
        
        # Search for each keyword
        for keyword, tickers in self.keyword_to_ticker.items():
            if keyword in text_lower:
                # Count occurrences
                count = text_lower.count(keyword)
                
                for ticker in tickers:
                    if ticker not in detected:
                        detected[ticker] = {
                            "ticker": ticker,
                            "matches": [],
                            "count": 0
                        }
                    detected[ticker]["matches"].append(keyword)
                    detected[ticker]["count"] += count
        
        # Calculate confidence scores
        results = []
        max_count = max([d["count"] for d in detected.values()]) if detected else 1
        
        for ticker, data in detected.items():
            confidence = min(1.0, data["count"] / 3)  # Normalize to 0-1
            
            if confidence >= min_confidence:
                results.append({
                    "ticker": ticker,
                    "matches": list(set(data["matches"])),
                    "count": data["count"],
                    "confidence": confidence
                })
        
        # Sort by confidence
        results.sort(key=lambda x: x["confidence"], reverse=True)
        
        return results
    
    def find_assets_in_article(
        self,
        article: Dict,
        fields: List[str] = None
    ) -> List[Dict]:
        """
        Find assets in an article
        
        Args:
            article: Article dictionary with title, content, etc.
            fields: Fields to search (default: title, content)
        
        Returns:
            List of detected assets
        """
        if fields is None:
            fields = ["title", "content"]
        
        # Combine text from all fields
        text = " ".join([
            str(article.get(field, ""))
            for field in fields
        ])
        
        return self.find_assets_in_text(text)
    
    def link_articles_to_assets(
        self,
        articles: List[Dict],
        min_confidence: float = 0.5
    ) -> Dict[str, List[Dict]]:
        """
        Link multiple articles to assets
        
        Args:
            articles: List of article dictionaries
            min_confidence: Minimum confidence threshold
        
        Returns:
            Dictionary mapping ticker -> list of linked articles
        """
        asset_articles = defaultdict(list)
        
        for idx, article in enumerate(articles):
            assets = self.find_assets_in_article(article)
            
            for asset in assets:
                if asset["confidence"] >= min_confidence:
                    asset_articles[asset["ticker"]].append({
                        "article_idx": idx,
                        "article_id": article.get("id"),
                        "title": article.get("title", "")[:100],
                        "confidence": asset["confidence"],
                        "matches": asset["matches"],
                        "sentiment": article.get("sentiment", {})
                    })
        
        return dict(asset_articles)
    
    def get_asset_sentiment(
        self,
        ticker: str,
        linked_articles: Dict[str, List[Dict]]
    ) -> Dict:
        """
        Calculate aggregate sentiment for an asset
        
        Args:
            ticker: Asset ticker
            linked_articles: Output from link_articles_to_assets
        
        Returns:
            Aggregated sentiment data
        """
        if ticker not in linked_articles:
            return {"ticker": ticker, "error": "No articles found"}
        
        articles = linked_articles[ticker]
        
        if not articles:
            return {"ticker": ticker, "error": "No articles found"}
        
        # Aggregate sentiment scores
        sentiment_scores = []
        sentiment_labels = []
        
        for art in articles:
            sentiment = art.get("sentiment", {})
            if "score" in sentiment:
                # Weight by confidence
                weight = art.get("confidence", 1.0)
                sentiment_scores.append(sentiment["score"] * weight)
            if "label" in sentiment:
                sentiment_labels.append(sentiment["label"])
        
        # Calculate weighted average
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        
        # Count labels
        label_counts = {
            "positive": sentiment_labels.count("positive"),
            "negative": sentiment_labels.count("negative"),
            "neutral": sentiment_labels.count("neutral")
        }
        
        # Determine dominant sentiment
        dominant_label = max(label_counts, key=label_counts.get)
        
        return {
            "ticker": ticker,
            "article_count": len(articles),
            "avg_sentiment": avg_sentiment,
            "sentiment_label": dominant_label,
            "label_distribution": label_counts,
            "keywords": self.ticker_map.get(ticker, [])
        }
    
    def get_all_tracked_tickers(self) -> List[str]:
        """Get list of all tracked tickers"""
        return list(self.ticker_map.keys())
    
    def add_ticker(self, ticker: str, keywords: List[str]):
        """Add a new ticker mapping"""
        self.ticker_map[ticker] = keywords
        for keyword in keywords:
            self.keyword_to_ticker[keyword.lower()].append(ticker)


# Example usage
if __name__ == "__main__":
    tracker = AssetTracker()
    
    # Test article
    test_article = {
        "title": "Apple announces new iPhone with AI features",
        "content": "Apple Inc (AAPL) unveiled its latest iPhone model with advanced AI capabilities. Tim Cook presented the device at Apple Park. The stock rose 3% on the news. Meanwhile, Microsoft and Google are also racing to add AI to their products."
    }
    
    assets = tracker.find_assets_in_article(test_article)
    
    print("Detected assets:")
    for asset in assets:
        print(f"  {asset['ticker']}: confidence={asset['confidence']:.2f}, matches={asset['matches']}")
