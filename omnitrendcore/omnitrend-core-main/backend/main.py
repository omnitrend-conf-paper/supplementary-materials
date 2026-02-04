#!/usr/bin/env python3
"""
Main pipeline for news trend analysis using TGN
"""

import argparse
import json
import os
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from ingestion.main_collector import NewsCollectionOrchestrator as NewsCollector
from nlp.processor import MultilingualNLPProcessor
from graph.build_graph import NewsGraphBuilder
from tgn.model import NewsTGN
from tgn.train import TGNTrainer
from tgn.train import TGNTrainer
from utils.trend_analyzer import TrendAnalyzer
from ingestion.config import Config

# Import NewsAPI collector
try:
    from newsapi_collector import NewsAPICollector
    NEWSAPI_AVAILABLE = True
except ImportError:
    NEWSAPI_AVAILABLE = False

# Import Financial modules
try:
    from financial_collector import FinancialCollector
    from financial.asset_tracker import AssetTracker
    from financial.financial_analyzer import FinancialAnalyzer
    FINANCIAL_AVAILABLE = True
except ImportError:
    FINANCIAL_AVAILABLE = False

def run_data_collection(args):
    """Step 1: Collect news data from RSS feeds"""
    print("=" * 60)
    print("STEP 1: Data Collection (RSS)")
    print("=" * 60)
    
    collector = NewsCollector()
    
    print("Collecting news articles from RSS feeds...")
    parallel_mode = 'async' if args.async_collection else True
    results = collector.run(parallel=parallel_mode, output_format='json')
    
    print(f"Collected {len(results['articles'])} articles")
    print()


def run_live_collection(args):
    """Collect real-time news from NewsAPI"""
    print("=" * 60)
    print("STEP: Live News Collection (NewsAPI)")
    print("=" * 60)
    
    if not NEWSAPI_AVAILABLE:
        print("Error: NewsAPI collector not available")
        return
    
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        print("Error: NEWSAPI_KEY environment variable not set")
        print("Get a free API key from https://newsapi.org/register")
        print("Then run: export NEWSAPI_KEY='your_key_here'")
        return
    
    collector = NewsAPICollector(api_key)
    
    # Collect from all categories
    print("Collecting headlines from all categories...")
    articles = collector.collect_by_categories(
        categories=["business", "technology", "science", "health", "sports", "entertainment"],
        country=args.country,
        max_per_category=args.max_articles // 6
    )
    
    # Also search for trending topics if query provided
    if args.query:
        print(f"Searching for: {args.query}...")
        search_articles = collector.search_everything(
            query=args.query,
            max_articles=args.max_articles
        )
        articles.extend(search_articles)
    
    print(f"Collected {len(articles)} articles")
    
    # Save to raw data
    output_path = Path(args.raw_data_dir) / "newsapi_live.json"
    collector.save_articles(articles, str(output_path))
    print()


def run_financial_collection(args):
    """Collect financial price data"""
    print("=" * 60)
    print("STEP: Financial Data Collection")
    print("=" * 60)
    
    if not FINANCIAL_AVAILABLE:
        print("Error: Financial modules not available")
        return
    
    collector = FinancialCollector()
    tickers = [t.strip() for t in args.tickers.split(",")]
    
    print(f"Collecting price data for {len(tickers)} assets...")
    results = {}
    
    for ticker in tickers:
        print(f"  Fetching {ticker}...")
        results[ticker] = collector.get_asset_data(ticker, args.days)
    
    # Save data
    output_path = Path(args.raw_data_dir) / "financial_data.json"
    collector.save_data(results, str(output_path))
    
    # Print summary
    print(f"\nCollected data for {len(results)} assets:")
    for ticker, data in results.items():
        if "error" not in data:
            price = data.get('latest_price', 0)
            change = data.get('price_change_pct', 0)
            print(f"  {ticker}: ${price:,.2f} ({change:+.2f}%)")
        else:
            print(f"  {ticker}: Error - {data['error']}")
    print()


def run_financial_analysis(args):
    """Run financial sentiment-price correlation analysis"""
    print("=" * 60)
    print("STEP: Financial Analysis")
    print("=" * 60)
    
    if not FINANCIAL_AVAILABLE:
        print("Error: Financial modules not available")
        return
    
    # Load processed articles
    processed_path = Path(args.processed_data_dir) / "processed_news.json"
    if not processed_path.exists():
        print(f"Error: No processed articles found at {processed_path}")
        print("Run --step process first")
        return
    
    with open(processed_path, 'r') as f:
        articles = json.load(f)
    
    print(f"Loaded {len(articles)} processed articles")
    
    # Load financial data
    financial_path = Path(args.raw_data_dir) / "financial_data.json"
    if not financial_path.exists():
        print(f"Error: No financial data found at {financial_path}")
        print("Run --step collect-finance first")
        return
    
    with open(financial_path, 'r') as f:
        financial_data = json.load(f)
    
    # Track assets in articles
    tracker = AssetTracker()
    linked = tracker.link_articles_to_assets(articles)
    
    print(f"Linked articles to {len(linked)} assets")
    
    # Analyze each ticker
    analyzer = FinancialAnalyzer()
    results = {}
    
    tickers = [t.strip() for t in args.tickers.split(",")]
    
    for ticker in tickers:
        if ticker not in financial_data:
            print(f"  {ticker}: No price data available")
            continue
        
        if ticker not in linked and ticker not in [t.replace("-USD", "") for t in linked]:
            # Try alternate format
            alt_ticker = f"{ticker}-USD"
            if alt_ticker not in linked:
                print(f"  {ticker}: No news articles linked")
                continue
            ticker = alt_ticker
        
        print(f"  Analyzing {ticker}...")
        
        price_data = financial_data.get(ticker, {}).get("prices", [])
        if not price_data:
            print(f"  {ticker}: No price history")
            continue
        
        # Build sentiment timeline from linked articles
        sentiment_data = []
        for art in linked.get(ticker, []):
            article_idx = art.get("article_idx", 0)
            if article_idx < len(articles):
                article = articles[article_idx]
                sentiment_data.append({
                    "timestamp": article.get("published_at", ""),
                    "sentiment_score": art.get("sentiment", {}).get("score", 0.5),
                    "article_count": 1
                })
        
        if not sentiment_data:
            print(f"  {ticker}: No sentiment data")
            continue
        
        # Align and correlate
        aligned = analyzer.align_sentiment_prices(sentiment_data, price_data)
        
        if len(aligned) < 3:
            print(f"  {ticker}: Not enough aligned data points")
            continue
        
        correlation = analyzer.calculate_correlation(aligned)
        signals = analyzer.generate_signals(aligned)
        backtest = analyzer.backtest_signals(signals)
        
        results[ticker] = {
            "article_count": len(linked.get(ticker, [])),
            "correlation": correlation,
            "backtest": backtest,
            "latest_signal": signals[-1] if signals else None
        }
    
    # Save and print results
    output_path = Path(args.output_dir) / "financial_analysis.json"
    analyzer.save_analysis(results, str(output_path))
    
    print("\n" + "=" * 60)
    print("FINANCIAL ANALYSIS RESULTS")
    print("=" * 60)
    
    for ticker, data in results.items():
        corr = data["correlation"]
        print(f"\n{ticker}:")
        print(f"  Articles: {data['article_count']}")
        print(f"  Correlation: {corr['correlation']:.3f} ({corr['correlation_strength']})")
        print(f"  Backtest Return: {data['backtest']['total_return_pct']:+.2f}%")
        print(f"  Buy & Hold Return: {data['backtest']['buy_hold_return_pct']:+.2f}%")
        if data.get('latest_signal'):
            print(f"  Latest Signal: {data['latest_signal']['signal']}")
    
    print(f"\nFull analysis saved to: {output_path}")
    print()

def run_nlp_processing(args):
    """Step 2: Process articles with NLP"""
    print("=" * 60)
    print("STEP 2: NLP Processing")
    print("=" * 60)
    
    # Load raw articles
    raw_files = list(Path(args.raw_data_dir).glob("*.json"))
    if not raw_files:
        print(f"No raw data found in {args.raw_data_dir}")
        return
    
    # Load all articles
    all_articles = []
    for file in raw_files:
        with open(file, 'r', encoding='utf-8') as f:
            articles = json.load(f)
            all_articles.extend(articles if isinstance(articles, list) else [articles])
    
    print(f"Loaded {len(all_articles)} articles")
    
    # Process with NLP
    processor = MultilingualNLPProcessor(device=args.device)
    
    output_path = Path(args.processed_data_dir) / "processed_news.json"
    processed = processor.process_batch(all_articles, output_path)
    
    print(f"Processed {len(processed)} articles")
    print()

def run_graph_building(args):
    """Step 3: Build temporal graph"""
    print("=" * 60)
    print("STEP 3: Graph Building")
    print("=" * 60)
    
    # Load processed articles
    processed_file = Path(args.processed_data_dir) / "processed_news.json"
    if not processed_file.exists():
        print(f"Processed data not found at {processed_file}")
        return
    
    with open(processed_file, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    
    # Build graph
    builder = NewsGraphBuilder(similarity_threshold=args.similarity_threshold)
    graph, node_features, edge_features, timestamps = builder.build_graph(articles)
    
    # Save graph
    builder.save_graph(args.processed_data_dir)
    
    # Print statistics
    stats = builder.get_statistics()
    print("\nGraph Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    print()

def run_training(args):
    """Step 4: Train TGN model"""
    print("=" * 60)
    print("STEP 4: Training TGN Model")
    print("=" * 60)
    
    # Create model
    model = NewsTGN(
        input_dim=385,  # 384 from embeddings + 1 sentiment
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout
    )
    
    # Create trainer
    trainer = TGNTrainer(
        model=model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=args.device
    )
    
    # Load graph data
    data = trainer.load_graph_data(args.processed_data_dir)
    
    # Train
    trainer.train(
        data=data,
        num_epochs=args.num_epochs,
        save_dir=args.model_dir
    )
    print()

def run_analysis(args):
    """Step 5: Analyze trends"""
    print("=" * 60)
    print("STEP 5: Trend Analysis")
    print("=" * 60)
    
    # Load model
    model = NewsTGN(
        input_dim=385,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads
    )
    
    trainer = TGNTrainer(model, device=args.device)
    
    model_path = Path(args.model_dir) / "best_model.pt"
    if not model_path.exists():
        print(f"Trained model not found at {model_path}")
        return
    
    trainer.load_checkpoint(model_path)
    
    # Load data
    data = trainer.load_graph_data(args.processed_data_dir)
    
    processed_file = Path(args.processed_data_dir) / "processed_news.json"
    with open(processed_file, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    
    # Analyze
    analyzer = TrendAnalyzer(model, device=args.device)
    results = analyzer.analyze_trends(data, articles, top_k=args.top_k)
    
    # Visualize
    analyzer.visualize_trends(results, args.output_dir)
    
    # Save report
    report_path = Path(args.output_dir) / "trend_analysis_report.json"
    analyzer.save_report(results, str(report_path))
    
    # Print results
    print("\n" + "=" * 60)
    print("TREND ANALYSIS RESULTS")
    print("=" * 60)
    
    print(f"\nTop {args.top_k} Trending Topics:")
    for i, (topic, score, count) in enumerate(results['trending_topics'], 1):
        print(f"  {i}. {topic}: score={score:.3f}, articles={count}")
    
    print(f"\nTop {args.top_k} Trending Articles:")
    for i, art in enumerate(results['trending_articles'], 1):
        print(f"  {i}. {art['title'][:60]}...")
        print(f"     Score: {art['trend_score']:.3f}, Topics: {', '.join(art['topics'][:3])}")
    
    print(f"\nStatistics:")
    stats = results
    print(f"  Total articles analyzed: {len(stats['trend_scores'])}")
    print(f"  Average trend score: {stats['trend_scores'].mean():.3f}")
    print(f"  High trend articles: {(stats['trend_classes'] == 2).sum()}")
    print(f"  Medium trend articles: {(stats['trend_classes'] == 1).sum()}")
    print(f"  Low trend articles: {(stats['trend_classes'] == 0).sum()}")
    
    print(f"\nVisualizations saved to: {args.output_dir}")
    print(f"Report saved to: {report_path}")
    print()

def main():
    parser = argparse.ArgumentParser(
        description="News Trend Analysis Pipeline using TGN"
    )
    
    # Pipeline step selection
    parser.add_argument(
        '--step',
        type=str,
        choices=['collect', 'collect-live', 'collect-finance', 'process', 'build', 'train', 'analyze', 'analyze-finance', 'infer', 'infer-finance', 'all'],
        default='all',
        help='Which pipeline step to run (collect-live=NewsAPI, collect-finance=stocks/crypto, infer-finance=financial analysis)'
    )
    
    # Data directories
    parser.add_argument('--raw-data-dir', type=str, default='data/raw')
    parser.add_argument('--processed-data-dir', type=str, default='data/processed')
    parser.add_argument('--model-dir', type=str, default='models')
    parser.add_argument('--output-dir', type=str, default='visualizations')
    
    # Model parameters
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--num-layers', type=int, default=3)
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # Training parameters
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu), auto-detect if not specified')
    
    # Graph building parameters
    parser.add_argument('--similarity-threshold', type=float, default=0.6,
                       help='Similarity threshold for connecting articles')
    
    # Analysis parameters
    parser.add_argument('--top-k', type=int, default=10,
                       help='Number of top trends to report')
    
    # NewsAPI parameters
    parser.add_argument('--country', type=str, default='us',
                       help='Country code for NewsAPI headlines')
    parser.add_argument('--query', type=str, default=None,
                       help='Search query for NewsAPI')
    parser.add_argument('--max-articles', type=int, default=100,
                       help='Max articles to collect from NewsAPI')
    
    # Financial parameters
    parser.add_argument('--tickers', type=str, default='AAPL,TSLA,NVDA,BTC,ETH',
                       help='Comma-separated tickers for financial analysis')
    parser.add_argument('--days', type=int, default=30,
                       help='Days of price history to fetch')
    
    # Collection mode
    parser.add_argument('--async-collection', action='store_true',
                       help='Use asynchronous collection (fastest)')
    parser.add_argument('--target', type=int, help='Target article count for collection')
    
    args = parser.parse_args()
    
    # Create directories
    Path(args.raw_data_dir).mkdir(parents=True, exist_ok=True)
    Path(args.processed_data_dir).mkdir(parents=True, exist_ok=True)
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run pipeline steps
    if args.step in ['collect', 'all']:
        if args.target:
            Config.DEFAULT_COLLECTION.target_article_count = args.target
        run_data_collection(args)
    
    if args.step == 'collect-live':
        run_live_collection(args)
    
    if args.step == 'collect-finance':
        run_financial_collection(args)
    
    if args.step in ['process', 'all']:
        run_nlp_processing(args)
    
    if args.step in ['build', 'all']:
        run_graph_building(args)
    
    if args.step in ['train', 'all']:
        run_training(args)
    
    if args.step in ['analyze', 'all']:
        run_analysis(args)
    
    if args.step == 'analyze-finance':
        run_financial_analysis(args)
    
    # Inference: collect-live -> process -> analyze (uses trained model)
    if args.step == 'infer':
        run_live_collection(args)
        run_nlp_processing(args)
        run_graph_building(args)
        run_analysis(args)
    
    # Financial inference: collect news + prices -> process -> analyze financial
    if args.step == 'infer-finance':
        run_live_collection(args)
        run_financial_collection(args)
        run_nlp_processing(args)
        run_financial_analysis(args)
    
    print("=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
