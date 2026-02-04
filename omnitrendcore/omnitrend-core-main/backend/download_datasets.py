#!/usr/bin/env python3
"""
Download and format news datasets for training:
- AG News (120K articles, 4 categories)
- BBC News (2,225 articles, 5 categories)
- Optional: MIND (Microsoft News, very large)

Usage:
    python download_datasets.py --dataset ag_news
    python download_datasets.py --dataset bbc
    python download_datasets.py --dataset all
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import random
import hashlib

# Try to import datasets library (Hugging Face)
try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: 'datasets' library not installed. Run: pip install datasets")

# Categories mapping
AG_NEWS_CATEGORIES = {
    0: "World",
    1: "Sports", 
    2: "Business",
    3: "Technology"
}

BBC_CATEGORIES = ["business", "entertainment", "politics", "sport", "tech"]


def generate_fake_timestamp(idx: int, total: int) -> str:
    """Generate fake timestamps spread over last 90 days"""
    base_date = datetime.now() - timedelta(days=90)
    offset_days = (idx / total) * 90
    offset_hours = random.randint(0, 23)
    offset_minutes = random.randint(0, 59)
    timestamp = base_date + timedelta(days=offset_days, hours=offset_hours, minutes=offset_minutes)
    return timestamp.isoformat()


def generate_article_id(text: str) -> str:
    """Generate unique article ID from content hash"""
    return hashlib.md5(text.encode()).hexdigest()[:12]


def download_ag_news(output_dir: Path, max_articles: int = 10000) -> int:
    """Download AG News dataset from Hugging Face"""
    print("\n" + "="*60)
    print("Downloading AG News Dataset")
    print("="*60)
    
    if not HF_AVAILABLE:
        print("Error: datasets library required. Run: pip install datasets")
        return 0
    
    # Load dataset
    print("Loading from Hugging Face...")
    dataset = load_dataset("ag_news", split="train")
    
    articles = []
    total = min(len(dataset), max_articles)
    
    print(f"Processing {total} articles...")
    for idx in range(total):
        item = dataset[idx]
        
        # Extract text and label
        text = item["text"]
        label = item["label"]
        
        # Split into title and content (AG News format: "Title. Content...")
        parts = text.split(". ", 1)
        title = parts[0] if len(parts) > 0 else text[:100]
        content = parts[1] if len(parts) > 1 else text
        
        article = {
            "id": generate_article_id(text),
            "title": title,
            "content": content[:2000],  # Limit content length
            "source": "AG News",
            "category": AG_NEWS_CATEGORIES.get(label, "Unknown"),
            "published_at": generate_fake_timestamp(idx, total),
            "url": f"https://ag-news.example.com/article/{idx}"
        }
        articles.append(article)
        
        if (idx + 1) % 2000 == 0:
            print(f"  Processed {idx + 1}/{total} articles...")
    
    # Save to file
    output_file = output_dir / "ag_news_raw.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Saved {len(articles)} articles to {output_file}")
    return len(articles)


def download_bbc_news(output_dir: Path) -> int:
    """Download BBC News dataset from Hugging Face"""
    print("\n" + "="*60)
    print("Downloading BBC News Dataset")
    print("="*60)
    
    if not HF_AVAILABLE:
        print("Error: datasets library required. Run: pip install datasets")
        return 0
    
    # Load dataset
    print("Loading from Hugging Face...")
    try:
        dataset = load_dataset("SetFit/bbc-news", split="train")
    except Exception as e:
        print(f"Error loading BBC dataset: {e}")
        print("Trying alternative source...")
        try:
            dataset = load_dataset("fancyzhx/bbc_news", split="train")
        except:
            print("Could not load BBC News dataset")
            return 0
    
    articles = []
    total = len(dataset)
    
    print(f"Processing {total} articles...")
    for idx in range(total):
        item = dataset[idx]
        
        # Handle different dataset formats
        if "text" in item:
            text = item["text"]
            label = item.get("label", 0)
            if isinstance(label, int) and label < len(BBC_CATEGORIES):
                category = BBC_CATEGORIES[label]
            else:
                category = str(label)
        else:
            text = item.get("content", str(item))
            category = item.get("category", "Unknown")
        
        # Split into title and content
        lines = text.strip().split("\n")
        title = lines[0][:200] if lines else text[:100]
        content = "\n".join(lines[1:]) if len(lines) > 1 else text
        
        article = {
            "id": generate_article_id(text),
            "title": title,
            "content": content[:2000],
            "source": "BBC News",
            "category": category,
            "published_at": generate_fake_timestamp(idx, total),
            "url": f"https://bbc-news.example.com/article/{idx}"
        }
        articles.append(article)
    
    # Save to file
    output_file = output_dir / "bbc_news_raw.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Saved {len(articles)} articles to {output_file}")
    return len(articles)


def download_mind_demo(output_dir: Path, max_articles: int = 5000) -> int:
    """Download MIND dataset demo (smaller subset)"""
    print("\n" + "="*60)
    print("Downloading MIND Dataset (Demo Subset)")
    print("="*60)
    
    if not HF_AVAILABLE:
        print("Error: datasets library required. Run: pip install datasets")
        return 0
    
    print("Loading MIND demo from Hugging Face...")
    try:
        # MIND dataset is large, use smaller subset
        dataset = load_dataset("mind", "small", split="train", trust_remote_code=True)
    except Exception as e:
        print(f"MIND dataset not available: {e}")
        print("MIND requires manual download from Microsoft Research.")
        print("Visit: https://msnews.github.io/")
        return 0
    
    articles = []
    total = min(len(dataset), max_articles)
    
    print(f"Processing {total} articles...")
    for idx in range(total):
        item = dataset[idx]
        
        article = {
            "id": item.get("news_id", generate_article_id(str(item))),
            "title": item.get("title", ""),
            "content": item.get("abstract", ""),
            "source": "Microsoft News",
            "category": item.get("category", "Unknown"),
            "published_at": generate_fake_timestamp(idx, total),
            "url": item.get("url", f"https://msn.com/article/{idx}")
        }
        articles.append(article)
    
    output_file = output_dir / "mind_raw.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Saved {len(articles)} articles to {output_file}")
    return len(articles)


def combine_datasets(output_dir: Path) -> int:
    """Combine all downloaded datasets into one file"""
    print("\n" + "="*60)
    print("Combining All Datasets")
    print("="*60)
    
    all_articles = []
    
    for filename in ["ag_news_raw.json", "bbc_news_raw.json", "mind_raw.json"]:
        filepath = output_dir / filename
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                articles = json.load(f)
                print(f"  Loaded {len(articles)} from {filename}")
                all_articles.extend(articles)
    
    # Shuffle articles
    random.shuffle(all_articles)
    
    # Re-assign IDs and timestamps after shuffling
    for idx, article in enumerate(all_articles):
        article["id"] = idx + 1
        article["published_at"] = generate_fake_timestamp(idx, len(all_articles))
    
    # Save combined dataset
    output_file = output_dir / "combined_news_raw.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_articles, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Combined {len(all_articles)} articles to {output_file}")
    return len(all_articles)


def main():
    parser = argparse.ArgumentParser(description="Download news datasets")
    parser.add_argument("--dataset", type=str, default="all",
                       choices=["ag_news", "bbc", "mind", "all"],
                       help="Dataset to download")
    parser.add_argument("--max-articles", type=int, default=10000,
                       help="Maximum articles to download per dataset")
    parser.add_argument("--output-dir", type=str, default="data/raw",
                       help="Output directory")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total_articles = 0
    
    if args.dataset in ["ag_news", "all"]:
        total_articles += download_ag_news(output_dir, args.max_articles)
    
    if args.dataset in ["bbc", "all"]:
        total_articles += download_bbc_news(output_dir)
    
    if args.dataset in ["mind", "all"]:
        total_articles += download_mind_demo(output_dir, args.max_articles)
    
    if args.dataset == "all" and total_articles > 0:
        combine_datasets(output_dir)
    
    print("\n" + "="*60)
    print(f"DOWNLOAD COMPLETE: {total_articles} total articles")
    print("="*60)
    print("\nNext steps:")
    print("  1. Process with NLP: python main.py --step process")
    print("  2. Build graph: python main.py --step build")
    print("  3. Train model: python main.py --step train --num-epochs 100")


if __name__ == "__main__":
    main()
