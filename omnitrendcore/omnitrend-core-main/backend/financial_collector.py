#!/usr/bin/env python3
"""
Financial Data Collector
Collects stock and crypto price data from various sources.

Sources:
- yfinance: Stocks, ETFs, Indices (free, unlimited)
- CoinGecko: Cryptocurrencies (free, 30 calls/min)

Usage:
    python financial_collector.py --ticker AAPL --days 30
    python financial_collector.py --ticker BTC-USD --days 7
    python financial_collector.py --crypto bitcoin --days 30
"""

import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union
import time

# Try importing yfinance
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("Warning: yfinance not installed. Run: pip install yfinance")

# Try importing pycoingecko
try:
    from pycoingecko import CoinGeckoAPI
    COINGECKO_AVAILABLE = True
except ImportError:
    COINGECKO_AVAILABLE = False
    print("Warning: pycoingecko not installed. Run: pip install pycoingecko")


class FinancialCollector:
    """Unified collector for stock and crypto price data"""
    
    # Common crypto ticker mappings (yfinance format -> CoinGecko ID)
    CRYPTO_MAP = {
        "BTC": "bitcoin",
        "ETH": "ethereum",
        "SOL": "solana",
        "XRP": "ripple",
        "ADA": "cardano",
        "DOGE": "dogecoin",
        "DOT": "polkadot",
        "LINK": "chainlink",
        "AVAX": "avalanche-2",
        "MATIC": "matic-network",
    }
    
    def __init__(self):
        self.cg = CoinGeckoAPI() if COINGECKO_AVAILABLE else None
    
    # ============== STOCK DATA (yfinance) ==============
    
    def get_stock_price(
        self,
        ticker: str,
        days: int = 30,
        interval: str = "1d"
    ) -> Dict:
        """
        Get historical stock price data
        
        Args:
            ticker: Stock ticker (e.g., AAPL, TSLA, MSFT)
            days: Number of days of history
            interval: Data interval (1m, 5m, 15m, 1h, 1d, 1wk, 1mo)
        
        Returns:
            Dictionary with OHLCV data
        """
        if not YFINANCE_AVAILABLE:
            return {"error": "yfinance not installed"}
        
        try:
            stock = yf.Ticker(ticker)
            
            # Calculate period
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Fetch data
            hist = stock.history(start=start_date, end=end_date, interval=interval)
            
            if hist.empty:
                return {"error": f"No data found for {ticker}"}
            
            # Get company info
            info = stock.info
            
            # Convert to dict
            data = {
                "ticker": ticker,
                "name": info.get("shortName", ticker),
                "currency": info.get("currency", "USD"),
                "exchange": info.get("exchange", ""),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "market_cap": info.get("marketCap", 0),
                "collected_at": datetime.now().isoformat(),
                "interval": interval,
                "prices": []
            }
            
            for idx, row in hist.iterrows():
                data["prices"].append({
                    "timestamp": idx.isoformat(),
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": int(row["Volume"]) if row["Volume"] else 0
                })
            
            # Add latest price info
            data["latest_price"] = data["prices"][-1]["close"] if data["prices"] else 0
            data["price_change_pct"] = self._calculate_change(data["prices"])
            
            return data
            
        except Exception as e:
            return {"error": str(e), "ticker": ticker}
    
    def get_multiple_stocks(
        self,
        tickers: List[str],
        days: int = 30,
        interval: str = "1d"
    ) -> Dict[str, Dict]:
        """Get data for multiple stocks"""
        results = {}
        for ticker in tickers:
            print(f"  Fetching {ticker}...")
            results[ticker] = self.get_stock_price(ticker, days, interval)
            time.sleep(0.1)  # Small delay to be nice to API
        return results
    
    # ============== CRYPTO DATA (CoinGecko) ==============
    
    def get_crypto_price(
        self,
        crypto_id: str,
        days: int = 30,
        vs_currency: str = "usd"
    ) -> Dict:
        """
        Get historical crypto price data from CoinGecko
        
        Args:
            crypto_id: CoinGecko coin ID (e.g., bitcoin, ethereum)
            days: Number of days of history
            vs_currency: Quote currency (usd, eur, etc.)
        
        Returns:
            Dictionary with price data
        """
        if not COINGECKO_AVAILABLE:
            return {"error": "pycoingecko not installed"}
        
        try:
            # Get market chart data
            market_data = self.cg.get_coin_market_chart_by_id(
                id=crypto_id,
                vs_currency=vs_currency,
                days=days
            )
            
            # Get coin info
            coin_info = self.cg.get_coin_by_id(
                id=crypto_id,
                localization=False,
                tickers=False,
                market_data=True,
                community_data=False,
                developer_data=False
            )
            
            data = {
                "crypto_id": crypto_id,
                "symbol": coin_info.get("symbol", crypto_id).upper(),
                "name": coin_info.get("name", crypto_id),
                "currency": vs_currency.upper(),
                "market_cap": coin_info.get("market_data", {}).get("market_cap", {}).get(vs_currency, 0),
                "collected_at": datetime.now().isoformat(),
                "prices": []
            }
            
            # Parse price data
            prices = market_data.get("prices", [])
            volumes = market_data.get("total_volumes", [])
            
            for i, (timestamp, price) in enumerate(prices):
                volume = volumes[i][1] if i < len(volumes) else 0
                data["prices"].append({
                    "timestamp": datetime.fromtimestamp(timestamp / 1000).isoformat(),
                    "price": float(price),
                    "volume": float(volume)
                })
            
            # Add latest info
            data["latest_price"] = data["prices"][-1]["price"] if data["prices"] else 0
            data["price_change_pct"] = self._calculate_crypto_change(data["prices"])
            
            return data
            
        except Exception as e:
            return {"error": str(e), "crypto_id": crypto_id}
    
    def get_multiple_cryptos(
        self,
        crypto_ids: List[str],
        days: int = 30,
        vs_currency: str = "usd"
    ) -> Dict[str, Dict]:
        """Get data for multiple cryptocurrencies"""
        results = {}
        for crypto_id in crypto_ids:
            print(f"  Fetching {crypto_id}...")
            results[crypto_id] = self.get_crypto_price(crypto_id, days, vs_currency)
            time.sleep(2)  # CoinGecko rate limit
        return results
    
    # ============== UNIFIED INTERFACE ==============
    
    def get_asset_data(
        self,
        asset: str,
        days: int = 30,
        asset_type: str = "auto"
    ) -> Dict:
        """
        Get data for any asset (stock or crypto)
        
        Args:
            asset: Ticker or crypto ID
            days: Days of history
            asset_type: "stock", "crypto", or "auto" (detect automatically)
        
        Returns:
            Unified price data dictionary
        """
        # Auto-detect asset type
        if asset_type == "auto":
            asset_upper = asset.upper()
            if asset_upper in self.CRYPTO_MAP:
                asset_type = "crypto"
                asset = self.CRYPTO_MAP[asset_upper]
            elif asset.lower() in ["bitcoin", "ethereum", "solana", "ripple", "dogecoin"]:
                asset_type = "crypto"
            elif "-USD" in asset_upper:
                asset_type = "stock"  # yfinance crypto format
            else:
                asset_type = "stock"
        
        if asset_type == "crypto":
            return self.get_crypto_price(asset, days)
        else:
            return self.get_stock_price(asset, days)
    
    # ============== UTILITIES ==============
    
    def _calculate_change(self, prices: List[Dict]) -> float:
        """Calculate percentage change from OHLCV data"""
        if len(prices) < 2:
            return 0.0
        first_close = prices[0]["close"]
        last_close = prices[-1]["close"]
        if first_close == 0:
            return 0.0
        return ((last_close - first_close) / first_close) * 100
    
    def _calculate_crypto_change(self, prices: List[Dict]) -> float:
        """Calculate percentage change from crypto price data"""
        if len(prices) < 2:
            return 0.0
        first_price = prices[0]["price"]
        last_price = prices[-1]["price"]
        if first_price == 0:
            return 0.0
        return ((last_price - first_price) / first_price) * 100
    
    def save_data(self, data: Union[Dict, List], output_path: str):
        """Save data to JSON file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"Saved data to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Collect financial data")
    parser.add_argument("--ticker", type=str, help="Stock ticker (e.g., AAPL, TSLA)")
    parser.add_argument("--crypto", type=str, help="Crypto ID (e.g., bitcoin, ethereum)")
    parser.add_argument("--tickers", type=str, help="Comma-separated tickers")
    parser.add_argument("--days", type=int, default=30, help="Days of history")
    parser.add_argument("--interval", type=str, default="1d", help="Data interval for stocks")
    parser.add_argument("--output", type=str, default="data/raw/financial_data.json")
    args = parser.parse_args()
    
    collector = FinancialCollector()
    
    if args.tickers:
        # Multiple tickers
        tickers = [t.strip() for t in args.tickers.split(",")]
        print(f"Collecting data for {len(tickers)} assets...")
        results = {}
        
        for ticker in tickers:
            print(f"  Fetching {ticker}...")
            results[ticker] = collector.get_asset_data(ticker, args.days)
        
        collector.save_data(results, args.output)
        
        # Print summary
        print(f"\nCollected data for {len(results)} assets:")
        for ticker, data in results.items():
            if "error" not in data:
                print(f"  {ticker}: ${data.get('latest_price', 0):.2f} ({data.get('price_change_pct', 0):+.2f}%)")
            else:
                print(f"  {ticker}: Error - {data['error']}")
    
    elif args.ticker:
        # Single stock
        print(f"Collecting stock data for {args.ticker}...")
        data = collector.get_stock_price(args.ticker, args.days, args.interval)
        
        if "error" not in data:
            collector.save_data(data, args.output)
            print(f"\n{data['name']} ({data['ticker']})")
            print(f"  Latest: ${data['latest_price']:.2f}")
            print(f"  Change: {data['price_change_pct']:+.2f}%")
            print(f"  Data points: {len(data['prices'])}")
        else:
            print(f"Error: {data['error']}")
    
    elif args.crypto:
        # Single crypto
        print(f"Collecting crypto data for {args.crypto}...")
        data = collector.get_crypto_price(args.crypto, args.days)
        
        if "error" not in data:
            collector.save_data(data, args.output)
            print(f"\n{data['name']} ({data['symbol']})")
            print(f"  Latest: ${data['latest_price']:.2f}")
            print(f"  Change: {data['price_change_pct']:+.2f}%")
            print(f"  Data points: {len(data['prices'])}")
        else:
            print(f"Error: {data['error']}")
    
    else:
        # Default: popular assets
        print("Collecting data for popular assets...")
        tickers = ["AAPL", "TSLA", "NVDA", "BTC", "ETH"]
        results = {}
        
        for ticker in tickers:
            print(f"  Fetching {ticker}...")
            results[ticker] = collector.get_asset_data(ticker, args.days)
            time.sleep(1)
        
        collector.save_data(results, args.output)
        
        print(f"\nCollected data for {len(results)} assets:")
        for ticker, data in results.items():
            if "error" not in data:
                price = data.get('latest_price', 0)
                change = data.get('price_change_pct', 0)
                print(f"  {ticker}: ${price:,.2f} ({change:+.2f}%)")
            else:
                print(f"  {ticker}: Error - {data['error']}")


if __name__ == "__main__":
    main()
