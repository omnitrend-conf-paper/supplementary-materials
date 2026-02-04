"""
Financial Analyzer - Sentiment-Price Correlation Analysis

Analyzes the relationship between news sentiment and asset price movements
to detect trading signals and predict market trends.
"""

import json
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


class FinancialAnalyzer:
    """Analyzes sentiment-price correlations for financial assets"""
    
    def __init__(self):
        self.correlation_cache = {}
    
    def align_sentiment_prices(
        self,
        sentiment_data: List[Dict],
        price_data: List[Dict],
        time_window_hours: int = 24
    ) -> List[Dict]:
        """
        Align sentiment timestamps with price data
        
        Args:
            sentiment_data: List of {timestamp, sentiment_score, article_count}
            price_data: List of {timestamp, close/price, volume}
            time_window_hours: Window to aggregate sentiment before price point
        
        Returns:
            Aligned data points with sentiment and price
        """
        aligned = []
        
        # Parse price timestamps
        price_points = []
        for p in price_data:
            try:
                if isinstance(p.get("timestamp"), str):
                    ts = datetime.fromisoformat(p["timestamp"].replace("Z", "+00:00"))
                    if ts.tzinfo:
                        ts = ts.replace(tzinfo=None)
                else:
                    ts = p["timestamp"]
                
                price_val = p.get("close") or p.get("price", 0)
                price_points.append({
                    "timestamp": ts,
                    "price": float(price_val),
                    "volume": p.get("volume", 0)
                })
            except:
                continue
        
        # Parse sentiment timestamps
        sentiment_points = []
        for s in sentiment_data:
            try:
                if isinstance(s.get("timestamp"), str):
                    ts = datetime.fromisoformat(s["timestamp"].replace("Z", "+00:00"))
                    if ts.tzinfo:
                        ts = ts.replace(tzinfo=None)
                else:
                    ts = s["timestamp"]
                
                sentiment_points.append({
                    "timestamp": ts,
                    "score": s.get("sentiment_score", s.get("score", 0)),
                    "count": s.get("article_count", s.get("count", 1))
                })
            except:
                continue
        
        # Sort by time
        price_points.sort(key=lambda x: x["timestamp"])
        sentiment_points.sort(key=lambda x: x["timestamp"])
        
        # For each price point, aggregate preceding sentiment
        window = timedelta(hours=time_window_hours)
        
        for pp in price_points:
            price_time = pp["timestamp"]
            window_start = price_time - window
            
            # Find sentiment in window
            window_sentiment = [
                s for s in sentiment_points
                if window_start <= s["timestamp"] <= price_time
            ]
            
            if window_sentiment:
                # Weighted average by article count
                total_weight = sum(s["count"] for s in window_sentiment)
                if total_weight > 0:
                    avg_sentiment = sum(s["score"] * s["count"] for s in window_sentiment) / total_weight
                else:
                    avg_sentiment = sum(s["score"] for s in window_sentiment) / len(window_sentiment)
                
                aligned.append({
                    "timestamp": price_time.isoformat(),
                    "price": pp["price"],
                    "volume": pp["volume"],
                    "sentiment_score": avg_sentiment,
                    "sentiment_count": len(window_sentiment),
                    "total_articles": sum(s["count"] for s in window_sentiment)
                })
        
        return aligned
    
    def calculate_correlation(
        self,
        aligned_data: List[Dict],
        lag_hours: int = 0
    ) -> Dict:
        """
        Calculate correlation between sentiment and price changes
        
        Args:
            aligned_data: Output from align_sentiment_prices
            lag_hours: Lag between sentiment and price (positive = sentiment leads)
        
        Returns:
            Correlation statistics
        """
        if len(aligned_data) < 3:
            return {"error": "Not enough data points", "correlation": 0}
        
        # Extract arrays
        sentiments = np.array([d["sentiment_score"] for d in aligned_data])
        prices = np.array([d["price"] for d in aligned_data])
        
        # Calculate price changes (returns)
        price_changes = np.diff(prices) / prices[:-1] * 100  # Percentage change
        
        # Align with sentiment (drop last sentiment for diff alignment)
        sentiment_aligned = sentiments[:-1]
        
        # Apply lag if specified
        if lag_hours > 0 and len(sentiment_aligned) > lag_hours:
            sentiment_aligned = sentiment_aligned[:-lag_hours]
            price_changes = price_changes[lag_hours:]
        
        if len(sentiment_aligned) < 2:
            return {"error": "Not enough aligned points", "correlation": 0}
        
        # Calculate Pearson correlation
        correlation = np.corrcoef(sentiment_aligned, price_changes)[0, 1]
        if np.isnan(correlation):
            correlation = 0
        
        # Calculate additional statistics
        sentiment_mean = float(np.mean(sentiments))
        sentiment_std = float(np.std(sentiments))
        price_change_mean = float(np.mean(price_changes))
        price_change_std = float(np.std(price_changes))
        
        return {
            "correlation": float(correlation),
            "correlation_strength": self._interpret_correlation(correlation),
            "data_points": len(sentiment_aligned),
            "lag_hours": lag_hours,
            "sentiment_stats": {
                "mean": sentiment_mean,
                "std": sentiment_std,
                "min": float(np.min(sentiments)),
                "max": float(np.max(sentiments))
            },
            "price_change_stats": {
                "mean": price_change_mean,
                "std": price_change_std,
                "min": float(np.min(price_changes)),
                "max": float(np.max(price_changes))
            }
        }
    
    def _interpret_correlation(self, corr: float) -> str:
        """Interpret correlation strength"""
        abs_corr = abs(corr)
        if abs_corr >= 0.7:
            strength = "strong"
        elif abs_corr >= 0.4:
            strength = "moderate"
        elif abs_corr >= 0.2:
            strength = "weak"
        else:
            strength = "negligible"
        
        direction = "positive" if corr >= 0 else "negative"
        return f"{strength}_{direction}"
    
    def find_optimal_lag(
        self,
        aligned_data: List[Dict],
        max_lag_hours: int = 48
    ) -> Dict:
        """
        Find the lag that maximizes correlation
        
        Args:
            aligned_data: Aligned sentiment-price data
            max_lag_hours: Maximum lag to test
        
        Returns:
            Best lag and correlation at each lag tested
        """
        results = []
        
        for lag in range(0, max_lag_hours + 1, 4):  # Test every 4 hours
            corr_data = self.calculate_correlation(aligned_data, lag_hours=lag)
            results.append({
                "lag_hours": lag,
                "correlation": corr_data.get("correlation", 0),
                "data_points": corr_data.get("data_points", 0)
            })
        
        # Find best lag
        if results:
            best = max(results, key=lambda x: abs(x["correlation"]))
        else:
            best = {"lag_hours": 0, "correlation": 0}
        
        return {
            "optimal_lag_hours": best["lag_hours"],
            "optimal_correlation": best["correlation"],
            "all_lags": results
        }
    
    def generate_signals(
        self,
        aligned_data: List[Dict],
        sentiment_threshold_buy: float = 0.6,
        sentiment_threshold_sell: float = 0.4
    ) -> List[Dict]:
        """
        Generate experimental trading signals based on sentiment
        
        Args:
            aligned_data: Aligned sentiment-price data
            sentiment_threshold_buy: Sentiment above this = potential buy
            sentiment_threshold_sell: Sentiment below this = potential sell
        
        Returns:
            List of signals with timestamps
        """
        signals = []
        
        for i, data in enumerate(aligned_data):
            sentiment = data["sentiment_score"]
            
            signal = None
            if sentiment >= sentiment_threshold_buy:
                signal = "BUY"
            elif sentiment <= sentiment_threshold_sell:
                signal = "SELL"
            else:
                signal = "HOLD"
            
            signals.append({
                "timestamp": data["timestamp"],
                "signal": signal,
                "sentiment_score": sentiment,
                "price": data["price"],
                "article_count": data.get("total_articles", 0)
            })
        
        return signals
    
    def backtest_signals(
        self,
        signals: List[Dict],
        initial_capital: float = 10000,
        position_size: float = 0.1
    ) -> Dict:
        """
        Backtest trading signals (simplified simulation)
        
        Args:
            signals: Output from generate_signals
            initial_capital: Starting capital
            position_size: Fraction of capital per trade
        
        Returns:
            Backtest results
        """
        capital = initial_capital
        position = 0  # Number of shares
        trades = []
        
        for signal in signals:
            price = signal["price"]
            
            if signal["signal"] == "BUY" and position == 0:
                # Buy
                shares_to_buy = (capital * position_size) / price
                position = shares_to_buy
                capital -= shares_to_buy * price
                trades.append({
                    "timestamp": signal["timestamp"],
                    "action": "BUY",
                    "price": price,
                    "shares": shares_to_buy
                })
            
            elif signal["signal"] == "SELL" and position > 0:
                # Sell
                capital += position * price
                trades.append({
                    "timestamp": signal["timestamp"],
                    "action": "SELL",
                    "price": price,
                    "shares": position
                })
                position = 0
        
        # Calculate final value
        final_price = signals[-1]["price"] if signals else 0
        final_value = capital + (position * final_price)
        
        total_return = ((final_value - initial_capital) / initial_capital) * 100
        
        # Compare to buy & hold
        if signals:
            buy_hold_return = ((signals[-1]["price"] - signals[0]["price"]) / signals[0]["price"]) * 100
        else:
            buy_hold_return = 0
        
        return {
            "initial_capital": initial_capital,
            "final_value": final_value,
            "total_return_pct": total_return,
            "buy_hold_return_pct": buy_hold_return,
            "outperformance": total_return - buy_hold_return,
            "num_trades": len(trades),
            "trades": trades
        }
    
    def analyze_asset(
        self,
        ticker: str,
        articles: List[Dict],
        price_data: List[Dict],
        linked_articles: Dict = None
    ) -> Dict:
        """
        Complete analysis for a single asset
        
        Args:
            ticker: Asset ticker
            articles: List of processed articles
            price_data: Price history
            linked_articles: Pre-computed article linkage (optional)
        
        Returns:
            Complete analysis results
        """
        from .asset_tracker import AssetTracker
        
        # Link articles if not provided
        if linked_articles is None:
            tracker = AssetTracker()
            linked_articles = tracker.link_articles_to_assets(articles)
        
        if ticker not in linked_articles:
            return {"ticker": ticker, "error": "No articles linked to this asset"}
        
        # Build sentiment timeline
        sentiment_timeline = []
        for art in linked_articles[ticker]:
            sentiment = art.get("sentiment", {})
            score = sentiment.get("score", 0.5)
            
            # Get article timestamp
            article_idx = art.get("article_idx", 0)
            if article_idx < len(articles):
                timestamp = articles[article_idx].get("published_at", "")
            else:
                continue
            
            sentiment_timeline.append({
                "timestamp": timestamp,
                "sentiment_score": score,
                "article_count": 1
            })
        
        if not sentiment_timeline:
            return {"ticker": ticker, "error": "No sentiment data"}
        
        # Align with prices
        aligned = self.align_sentiment_prices(sentiment_timeline, price_data)
        
        if not aligned:
            return {"ticker": ticker, "error": "Could not align data"}
        
        # Calculate correlation
        correlation = self.calculate_correlation(aligned)
        
        # Find optimal lag
        optimal_lag = self.find_optimal_lag(aligned)
        
        # Generate signals
        signals = self.generate_signals(aligned)
        
        # Backtest
        backtest = self.backtest_signals(signals)
        
        return {
            "ticker": ticker,
            "article_count": len(linked_articles[ticker]),
            "aligned_data_points": len(aligned),
            "correlation": correlation,
            "optimal_lag": optimal_lag,
            "backtest": backtest,
            "latest_signal": signals[-1] if signals else None
        }
    
    def save_analysis(self, results: Dict, output_path: str):
        """Save analysis results to JSON"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"Saved analysis to {output_path}")


# Example usage
if __name__ == "__main__":
    analyzer = FinancialAnalyzer()
    
    # Test with dummy data
    sentiment_data = [
        {"timestamp": "2024-01-01T10:00:00", "sentiment_score": 0.6, "article_count": 5},
        {"timestamp": "2024-01-02T10:00:00", "sentiment_score": 0.7, "article_count": 8},
        {"timestamp": "2024-01-03T10:00:00", "sentiment_score": 0.5, "article_count": 3},
        {"timestamp": "2024-01-04T10:00:00", "sentiment_score": 0.4, "article_count": 6},
        {"timestamp": "2024-01-05T10:00:00", "sentiment_score": 0.8, "article_count": 10},
    ]
    
    price_data = [
        {"timestamp": "2024-01-01T16:00:00", "close": 100, "volume": 1000000},
        {"timestamp": "2024-01-02T16:00:00", "close": 102, "volume": 1200000},
        {"timestamp": "2024-01-03T16:00:00", "close": 101, "volume": 900000},
        {"timestamp": "2024-01-04T16:00:00", "close": 99, "volume": 1500000},
        {"timestamp": "2024-01-05T16:00:00", "close": 105, "volume": 2000000},
    ]
    
    # Align data
    aligned = analyzer.align_sentiment_prices(sentiment_data, price_data)
    print(f"Aligned {len(aligned)} data points")
    
    # Calculate correlation
    corr = analyzer.calculate_correlation(aligned)
    print(f"Correlation: {corr['correlation']:.3f} ({corr['correlation_strength']})")
    
    # Generate signals
    signals = analyzer.generate_signals(aligned)
    print(f"Generated {len(signals)} signals")
