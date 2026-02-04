#!/usr/bin/env python3
"""
Full Benchmark Runner
Runs all baseline models on the full dataset and generates comparison report.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import json
import numpy as np
from tgn.model import NewsTGN
from tgn.train import TGNTrainer
from benchmarks.compare import BenchmarkRunner
from benchmarks.evaluate import Evaluator


def main():
    print("=" * 60)
    print("FULL BENCHMARK COMPARISON")
    print("=" * 60)
    
    # Load graph data using trainer's method (handles edge format properly)
    print("\n1. Loading graph data...")
    data = TGNTrainer.load_graph_data("data/processed")
    print(f"   Nodes: {data.x.shape[0]}")
    print(f"   Features: {data.x.shape[1]}")
    print(f"   Edges: {data.edge_index.shape[1]}")
    
    # Load article texts for TF-IDF baseline
    print("\n2. Loading article texts...")
    with open("data/processed/processed_news.json", 'r') as f:
        articles = json.load(f)
    
    num_nodes = data.x.shape[0]
    texts = []
    for i, a in enumerate(articles[:num_nodes]):
        title = a.get('title') or ''
        content = a.get('content') or ''
        text = title + ' ' + content[:500]
        texts.append(text)
    print(f"   Loaded {len(texts)} article texts")
    
    # Load timestamps for Recency baseline
    print("\n3. Loading timestamps...")
    with open("data/processed/timestamps.json", 'r') as f:
        timestamps_dict = json.load(f)
    timestamps = np.array(list(timestamps_dict.values()))[:num_nodes]
    print(f"   Loaded {len(timestamps)} timestamps")
    
    # Load trained model (try optimized first)
    print("\n4. Loading trained TGN model...")
    model = NewsTGN(
        input_dim=data.x.shape[1],
        hidden_dim=512,
        num_layers=3,
        num_heads=8  # More heads for optimized model
    )
    trainer = TGNTrainer(model, device='cuda')
    
    # Try optimized model first
    optimized_path = Path("models/optimized_model.pt")
    best_path = Path("models/best_model.pt")
    
    if optimized_path.exists():
        checkpoint = torch.load(optimized_path, map_location='cuda')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"   Loaded OPTIMIZED model from {optimized_path}")
        print(f"   Best training accuracy: {checkpoint.get('best_accuracy', 'N/A')}")
    else:
        trainer.load_checkpoint(best_path)
        print(f"   Loaded model from {best_path}")
    
    # Use the same labels that the model was trained on
    # data.y is already set by load_graph_data() using degree-based pseudo-labels
    print(f"   Labels distribution: Low={int((data.y==0).sum())}, Med={int((data.y==1).sum())}, High={int((data.y==2).sum())}")
    
    # Run benchmarks
    print("\n5. Running benchmarks (this may take 5-10 minutes)...")
    runner = BenchmarkRunner(device='cuda')
    
    results = runner.run_all_benchmarks(
        data=data,
        texts=texts,
        timestamps=timestamps,
        proposed_model=trainer.model,
        num_epochs=100  # Full training
    )
    
    # Generate report
    print("\n6. Generating report...")
    report_path = runner.generate_report("visualizations")
    
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print("\n" + Evaluator.format_results_table(results))
    
    print(f"\nReport saved to: {report_path}")
    print(f"Chart saved to: visualizations/benchmark_comparison.png")
    print(f"JSON saved to: visualizations/benchmark_results.json")


if __name__ == "__main__":
    main()
