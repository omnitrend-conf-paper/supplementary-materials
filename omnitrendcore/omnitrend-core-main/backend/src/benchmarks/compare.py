"""
Benchmark Runner and Comparison Report Generator

Runs all baseline methods and the proposed TGN on the same data,
computes metrics, and generates comparison reports.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

from .baselines import (
    TFIDFBaseline,
    MLPBaseline,
    StaticGCNBaseline,
    StandardTGNBaseline,
    RecencyBaseline,
    DegreeBaseline,
    LSTMBaseline,
    GATBaseline,
    SVMBaseline,
    RandomForestBaseline
)
from .evaluate import Evaluator


class BenchmarkRunner:
    """Runs all benchmark methods and generates comparison report"""
    
    def __init__(self, device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.results = {}
        
    def train_nn_model(
        self,
        model: nn.Module,
        data,
        num_epochs: int = 50,
        lr: float = 1e-3
    ) -> nn.Module:
        """Train a neural network model"""
        model = model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        data = data.to(self.device)
        
        model.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            # Forward pass - different models have different signatures
            if isinstance(model, (MLPBaseline, LSTMBaseline)):
                output = model(data.x)
            else:
                output = model(data.x, data.edge_index, 
                             time_diffs=getattr(data, 'time_diffs', None))
            
            # Loss
            loss = nn.CrossEntropyLoss()(output['trend_class'], data.y)
            loss.backward()
            optimizer.step()
        
        return model
    
    def evaluate_model(
        self,
        model: nn.Module,
        data,
        is_mlp: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate a trained model"""
        model.eval()
        data = data.to(self.device)
        
        with torch.no_grad():
            if is_mlp:
                output = model(data.x)
            else:
                output = model(data.x, data.edge_index,
                             time_diffs=getattr(data, 'time_diffs', None))
        
        pred_class = output['trend_class'].argmax(dim=1).cpu().numpy()
        pred_score = output['trend_score'].squeeze().cpu().numpy()
        
        return pred_class, pred_score
    
    def run_all_benchmarks(
        self,
        data,
        texts: List[str] = None,
        timestamps: np.ndarray = None,
        proposed_model = None,
        num_epochs: int = 50
    ) -> Dict[str, Dict[str, float]]:
        """Run all benchmark methods"""
        
        y_true_class = data.y.cpu().numpy()
        y_true_score = y_true_class / 2.0  # Normalize to [0, 1]
        
        input_dim = data.x.shape[1]
        num_nodes = data.x.shape[0]
        
        print("Running benchmarks...")
        
        # 1. Recency Baseline
        print("  - Recency Baseline")
        if timestamps is not None:
            recency = RecencyBaseline()
            pred_class, pred_score = recency.predict(timestamps)
            self.results['Recency'] = Evaluator.full_evaluation(
                y_true_class, pred_class, y_true_score, pred_score
            )
        
        # 2. Degree Baseline
        print("  - Degree Baseline")
        degree = DegreeBaseline()
        pred_class, pred_score = degree.predict(data.edge_index, num_nodes)
        self.results['Degree'] = Evaluator.full_evaluation(
            y_true_class, pred_class, y_true_score, pred_score
        )
        
        # 3. TF-IDF Baseline (if texts provided)
        if texts is not None:
            print("  - TF-IDF Baseline")
            tfidf = TFIDFBaseline()
            pred_class, pred_score = tfidf.fit_predict(texts, y_true_class)
            self.results['TF-IDF'] = Evaluator.full_evaluation(
                y_true_class, pred_class, y_true_score, pred_score
            )
        
        # 4. MLP Baseline
        print("  - MLP Baseline")
        mlp = MLPBaseline(input_dim=input_dim)
        mlp = self.train_nn_model(mlp, data, num_epochs=num_epochs)
        pred_class, pred_score = self.evaluate_model(mlp, data, is_mlp=True)
        self.results['MLP'] = Evaluator.full_evaluation(
            y_true_class, pred_class, y_true_score, pred_score
        )
        
        # 5. Static GCN Baseline
        print("  - Static GCN Baseline")
        gcn = StaticGCNBaseline(input_dim=input_dim)
        gcn = self.train_nn_model(gcn, data, num_epochs=num_epochs)
        pred_class, pred_score = self.evaluate_model(gcn, data)
        self.results['Static GCN'] = Evaluator.full_evaluation(
            y_true_class, pred_class, y_true_score, pred_score
        )
        
        # 6. Standard TGN Baseline
        print("  - Standard TGN Baseline")
        std_tgn = StandardTGNBaseline(input_dim=input_dim)
        std_tgn = self.train_nn_model(std_tgn, data, num_epochs=num_epochs)
        pred_class, pred_score = self.evaluate_model(std_tgn, data)
        self.results['Standard TGN'] = Evaluator.full_evaluation(
            y_true_class, pred_class, y_true_score, pred_score
        )
        
        # 7. LSTM Baseline
        print("  - LSTM Baseline")
        lstm = LSTMBaseline(input_dim=input_dim)
        lstm = self.train_nn_model(lstm, data, num_epochs=num_epochs)
        pred_class, pred_score = self.evaluate_model(lstm, data, is_mlp=True)
        self.results['LSTM'] = Evaluator.full_evaluation(
            y_true_class, pred_class, y_true_score, pred_score
        )
        
        # 8. GAT Baseline
        print("  - GAT Baseline")
        gat = GATBaseline(input_dim=input_dim)
        gat = self.train_nn_model(gat, data, num_epochs=num_epochs)
        pred_class, pred_score = self.evaluate_model(gat, data)
        self.results['GAT'] = Evaluator.full_evaluation(
            y_true_class, pred_class, y_true_score, pred_score
        )
        
        # 9. SVM Baseline (if small enough dataset)
        if num_nodes < 20000:
            print("  - SVM Baseline")
            svm = SVMBaseline()
            features = data.x.cpu().numpy()
            pred_class, pred_score = svm.fit_predict(features, y_true_class)
            self.results['SVM'] = Evaluator.full_evaluation(
                y_true_class, pred_class, y_true_score, pred_score
            )
        
        # 10. Random Forest Baseline
        print("  - Random Forest Baseline")
        rf = RandomForestBaseline()
        features = data.x.cpu().numpy()
        pred_class, pred_score = rf.fit_predict(features, y_true_class)
        self.results['Random Forest'] = Evaluator.full_evaluation(
            y_true_class, pred_class, y_true_score, pred_score
        )
        
        # 7. Proposed Method (Advanced TGN)
        if proposed_model is not None:
            print("  - Proposed Method (Advanced TGN)")
            proposed_model.eval()
            proposed_model.to(self.device)
            data = data.to(self.device)
            
            with torch.no_grad():
                output = proposed_model(
                    data.x, data.edge_index,
                    time_diffs=getattr(data, 'time_diffs', None)
                )
            
            pred_class = output['trend_class'].argmax(dim=1).cpu().numpy()
            pred_score = output['trend_score'].squeeze().cpu().numpy()
            
            self.results['Proposed (Ours)'] = Evaluator.full_evaluation(
                y_true_class, pred_class, y_true_score, pred_score
            )
        
        print("Benchmarks completed!")
        return self.results
    
    def generate_report(self, output_dir: str = "visualizations") -> str:
        """Generate comparison report"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create markdown report
        report = f"""# Benchmark Comparison Report
Generated: {datetime.now().isoformat()}

## Results Summary

{Evaluator.format_results_table(self.results)}

## Key Findings

"""
        # Find best method for each metric
        if self.results:
            first_method = next(iter(self.results.values()))
            for metric in first_method.keys():
                values = {m: self.results[m][metric] for m in self.results}
                best = max(values.items(), key=lambda x: x[1])
                report += f"- **Best {metric}**: {best[0]} ({best[1]:.4f})\n"
        
        # Save report
        report_path = output_dir / "benchmark_report.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        # Save JSON results
        json_path = output_dir / "benchmark_results.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Generate comparison chart
        self._plot_comparison_chart(output_dir)
        
        return str(report_path)
    
    def _plot_comparison_chart(self, output_dir: Path):
        """Generate comparison bar chart"""
        if not self.results:
            return
        
        methods = list(self.results.keys())
        metrics = ['accuracy', 'f1_macro', 'spearman']
        
        x = np.arange(len(methods))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, metric in enumerate(metrics):
            values = [self.results[m].get(metric, 0) for m in methods]
            ax.bar(x + i * width, values, width, label=metric)
        
        ax.set_xlabel('Method')
        ax.set_ylabel('Score')
        ax.set_title('Benchmark Comparison')
        ax.set_xticks(x + width)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "benchmark_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()


# CLI interface
if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    
    from tgn.model import NewsTGN
    from tgn.train import TGNTrainer
    
    # Load data
    trainer = TGNTrainer(NewsTGN(input_dim=385, hidden_dim=256))
    data = trainer.load_graph_data("data/processed")
    
    # Load processed articles for texts
    with open("data/processed/processed_news.json", 'r') as f:
        articles = json.load(f)
    texts = [a['title'] + ' ' + a.get('content', '')[:500] for a in articles]
    
    # Load timestamps
    with open("data/processed/timestamps.json", 'r') as f:
        timestamps = np.array(list(json.load(f).values()))
    
    # Load trained model
    trainer.load_checkpoint(Path("models/best_model.pt"))
    
    # Run benchmarks
    runner = BenchmarkRunner()
    results = runner.run_all_benchmarks(
        data=data,
        texts=texts,
        timestamps=timestamps,
        proposed_model=trainer.model,
        num_epochs=50
    )
    
    # Generate report
    report_path = runner.generate_report()
    print(f"\nReport saved to: {report_path}")
    print("\n" + Evaluator.format_results_table(results))
