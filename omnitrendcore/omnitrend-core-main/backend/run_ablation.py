#!/usr/bin/env python3
"""
Ablation Study and Visualization Generator

Creates:
1. Ablation study - tests model with different components removed
2. Confusion matrix
3. t-SNE embeddings visualization
4. Training curves
5. Benchmark comparison chart
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
from datetime import datetime

from tgn.model import NewsTGN, TGNEncoder, TrendPredictor
from tgn.train import TGNTrainer


class AblationStudy:
    """Run ablation experiments to understand component contributions"""
    
    def __init__(self, data, device='cuda'):
        self.data = data.to(device)
        self.device = device
        self.results = {}
        
    def train_model(self, model, num_epochs=50, lr=5e-4):
        """Train a model and return accuracy"""
        model = model.to(self.device)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            output = model(
                self.data.x, 
                self.data.edge_index,
                time_diffs=getattr(self.data, 'time_diffs', None)
            )
            loss = criterion(output['trend_class'], self.data.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            output = model(
                self.data.x, 
                self.data.edge_index,
                time_diffs=getattr(self.data, 'time_diffs', None)
            )
            pred = output['trend_class'].argmax(dim=1)
            acc = (pred == self.data.y).float().mean().item()
        
        return acc, model, pred
    
    def run_ablation(self):
        """Run all ablation experiments"""
        input_dim = self.data.x.shape[1]
        
        print("=" * 60)
        print("ABLATION STUDY")
        print("=" * 60)
        
        # 1. Full model (512 dim, temporal, graph)
        print("\n1. Full Model (512 dim, temporal, graph)...")
        full_model = NewsTGN(input_dim=input_dim, hidden_dim=512, num_layers=3, num_heads=8)
        acc, model, pred_full = self.train_model(full_model, num_epochs=50)
        self.results['Full Model'] = acc
        print(f"   Accuracy: {acc:.4f}")
        
        # 2. Smaller model (256 dim)
        print("\n2. Smaller Model (256 dim)...")
        small_model = NewsTGN(input_dim=input_dim, hidden_dim=256, num_layers=3, num_heads=4)
        acc, _, _ = self.train_model(small_model, num_epochs=50)
        self.results['Smaller (256)'] = acc
        print(f"   Accuracy: {acc:.4f}")
        
        # 3. Tiny model (128 dim)
        print("\n3. Tiny Model (128 dim)...")
        tiny_model = NewsTGN(input_dim=input_dim, hidden_dim=128, num_layers=2, num_heads=4)
        acc, _, _ = self.train_model(tiny_model, num_epochs=50)
        self.results['Tiny (128)'] = acc
        print(f"   Accuracy: {acc:.4f}")
        
        # 4. Without temporal memory
        print("\n4. Without Temporal Memory...")
        no_mem_encoder = TGNEncoder(
            input_dim=input_dim, hidden_dim=512, num_layers=3, num_heads=8,
            use_temporal_memory=False
        )
        no_mem_model = nn.Sequential()
        no_mem_model.encoder = no_mem_encoder
        no_mem_model.predictor = TrendPredictor(hidden_dim=512)
        
        # Custom forward for this ablation
        class NoMemModel(nn.Module):
            def __init__(self, encoder, predictor):
                super().__init__()
                self.encoder = encoder
                self.predictor = predictor
            def forward(self, x, edge_index, time_diffs=None, **kwargs):
                enc_out = self.encoder(x, edge_index, time_diffs=time_diffs)
                trend_class, trend_score = self.predictor(enc_out['embeddings'])
                return {'trend_class': trend_class, 'trend_score': trend_score}
        
        no_mem = NoMemModel(no_mem_encoder, TrendPredictor(512))
        acc, _, _ = self.train_model(no_mem, num_epochs=50)
        self.results['No Temporal Memory'] = acc
        print(f"   Accuracy: {acc:.4f}")
        
        # 5. Fewer layers (1 layer)
        print("\n5. Single Layer...")
        single_layer = NewsTGN(input_dim=input_dim, hidden_dim=512, num_layers=1, num_heads=8)
        acc, _, _ = self.train_model(single_layer, num_epochs=50)
        self.results['Single Layer'] = acc
        print(f"   Accuracy: {acc:.4f}")
        
        # Clear GPU memory
        del single_layer
        torch.cuda.empty_cache()
        
        # 6. More layers (4 layers with smaller batch)
        print("\n6. Deep Model (4 layers)...")
        try:
            deep_model = NewsTGN(input_dim=input_dim, hidden_dim=256, num_layers=4, num_heads=4)
            acc, _, _ = self.train_model(deep_model, num_epochs=30)
            self.results['Deep (4 layers)'] = acc
            print(f"   Accuracy: {acc:.4f}")
            del deep_model
            torch.cuda.empty_cache()
        except RuntimeError as e:
            print(f"   Skipped due to memory: {e}")
            self.results['Deep (4 layers)'] = 0.0
        
        return self.results, pred_full
    
    def plot_ablation_results(self, output_dir):
        """Plot ablation study results"""
        output_dir = Path(output_dir)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        methods = list(self.results.keys())
        accuracies = list(self.results.values())
        colors = ['#2ecc71' if m == 'Full Model' else '#3498db' for m in methods]
        
        bars = ax.barh(methods, accuracies, color=colors)
        ax.set_xlabel('Accuracy', fontsize=12)
        ax.set_title('Ablation Study Results', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1)
        
        # Add value labels
        for bar, acc in zip(bars, accuracies):
            ax.text(acc + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{acc:.3f}', va='center', fontsize=10)
        
        # Highlight full model
        ax.axvline(x=self.results.get('Full Model', 0), color='#2ecc71', 
                   linestyle='--', alpha=0.7, label='Full Model')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'ablation_study.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir / 'ablation_study.png'}")


class Visualizer:
    """Generate visualizations for the report"""
    
    def __init__(self, data, model, device='cuda'):
        self.data = data.to(device)
        self.model = model.to(device)
        self.device = device
        
    def plot_confusion_matrix(self, y_true, y_pred, output_dir):
        """Plot confusion matrix"""
        output_dir = Path(output_dir)
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Low', 'Medium', 'High'],
                   yticklabels=['Low', 'Medium', 'High'])
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_title('Confusion Matrix - Proposed TGN', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir / 'confusion_matrix.png'}")
        
    def plot_tsne_embeddings(self, output_dir, n_samples=2000):
        """Plot t-SNE visualization of embeddings"""
        output_dir = Path(output_dir)
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(
                self.data.x, 
                self.data.edge_index,
                time_diffs=getattr(self.data, 'time_diffs', None)
            )
            embeddings = output['embeddings'].cpu().numpy()
        
        # Sample for faster t-SNE
        n_samples = min(n_samples, len(embeddings))
        indices = np.random.choice(len(embeddings), n_samples, replace=False)
        sampled_embeddings = embeddings[indices]
        sampled_labels = self.data.y.cpu().numpy()[indices]
        
        print("Running t-SNE (this may take a minute)...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d = tsne.fit_transform(sampled_embeddings)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = ['#e74c3c', '#f39c12', '#2ecc71']
        labels = ['Low', 'Medium', 'High']
        
        for i, (color, label) in enumerate(zip(colors, labels)):
            mask = sampled_labels == i
            ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                      c=color, label=label, alpha=0.6, s=30)
        
        ax.set_xlabel('t-SNE 1', fontsize=12)
        ax.set_ylabel('t-SNE 2', fontsize=12)
        ax.set_title('t-SNE Visualization of TGN Embeddings', fontsize=14, fontweight='bold')
        ax.legend(title='Trend Level')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'tsne_embeddings.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir / 'tsne_embeddings.png'}")
    
    def plot_benchmark_comparison(self, benchmark_results, output_dir):
        """Create improved benchmark comparison chart"""
        output_dir = Path(output_dir)
        
        methods = list(benchmark_results.keys())
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot 1: Accuracy
        accs = [benchmark_results[m]['accuracy'] for m in methods]
        colors = ['#2ecc71' if 'Proposed' in m else '#3498db' for m in methods]
        axes[0].barh(methods, accs, color=colors)
        axes[0].set_xlabel('Accuracy')
        axes[0].set_title('Classification Accuracy', fontweight='bold')
        axes[0].set_xlim(0, 1)
        
        # Plot 2: F1 Score
        f1s = [benchmark_results[m]['f1_macro'] for m in methods]
        axes[1].barh(methods, f1s, color=colors)
        axes[1].set_xlabel('F1 Macro')
        axes[1].set_title('F1 Score (Macro)', fontweight='bold')
        axes[1].set_xlim(0, 1)
        
        # Plot 3: Correlation
        corrs = [benchmark_results[m]['spearman'] for m in methods]
        axes[2].barh(methods, corrs, color=colors)
        axes[2].set_xlabel('Spearman Correlation')
        axes[2].set_title('Ranking Correlation', fontweight='bold')
        axes[2].set_xlim(-0.5, 1)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'benchmark_detailed.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir / 'benchmark_detailed.png'}")
    
    def plot_class_distribution(self, output_dir):
        """Plot class distribution"""
        output_dir = Path(output_dir)
        
        y = self.data.y.cpu().numpy()
        counts = [np.sum(y == i) for i in range(3)]
        
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(['Low', 'Medium', 'High'], counts, 
                     color=['#e74c3c', '#f39c12', '#2ecc71'])
        
        ax.set_xlabel('Trend Level', fontsize=12)
        ax.set_ylabel('Number of Articles', fontsize=12)
        ax.set_title('Class Distribution in Dataset', fontsize=14, fontweight='bold')
        
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                   f'{count}\n({count/sum(counts)*100:.1f}%)', 
                   ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'class_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir / 'class_distribution.png'}")


def main():
    print("=" * 60)
    print("ABLATION STUDY & VISUALIZATION GENERATOR")
    print("=" * 60)
    
    # Output directory
    output_dir = Path("visualizations")
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    print("\n1. Loading data...")
    data = TGNTrainer.load_graph_data("data/processed")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Nodes: {data.x.shape[0]}, Device: {device}")
    
    # Run ablation study
    print("\n2. Running ablation study...")
    ablation = AblationStudy(data, device)
    ablation_results, pred_full = ablation.run_ablation()
    ablation.plot_ablation_results(output_dir)
    
    # Save ablation results
    with open(output_dir / 'ablation_results.json', 'w') as f:
        json.dump(ablation_results, f, indent=2)
    print(f"\nSaved: {output_dir / 'ablation_results.json'}")
    
    # Load optimized model for visualizations
    print("\n3. Loading optimized model for visualizations...")
    model = NewsTGN(input_dim=data.x.shape[1], hidden_dim=512, num_layers=3, num_heads=8)
    checkpoint = torch.load("models/optimized_model.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Generate visualizations
    print("\n4. Generating visualizations...")
    viz = Visualizer(data, model, device)
    
    # Confusion matrix
    model.eval()
    with torch.no_grad():
        output = model(data.x.to(device), data.edge_index.to(device),
                      time_diffs=getattr(data, 'time_diffs', None).to(device) if hasattr(data, 'time_diffs') else None)
        pred = output['trend_class'].argmax(dim=1).cpu().numpy()
    
    viz.plot_confusion_matrix(data.y.cpu().numpy(), pred, output_dir)
    viz.plot_tsne_embeddings(output_dir)
    viz.plot_class_distribution(output_dir)
    
    # Load benchmark results for comparison chart
    benchmark_path = output_dir / 'benchmark_results.json'
    if benchmark_path.exists():
        with open(benchmark_path, 'r') as f:
            benchmark_results = json.load(f)
        viz.plot_benchmark_comparison(benchmark_results, output_dir)
    
    # Summary
    print("\n" + "=" * 60)
    print("ABLATION STUDY SUMMARY")
    print("=" * 60)
    print("\n| Configuration | Accuracy |")
    print("|---------------|----------|")
    for config, acc in sorted(ablation_results.items(), key=lambda x: x[1], reverse=True):
        marker = "⭐" if config == "Full Model" else ""
        print(f"| {config} | {acc:.4f} {marker}|")
    
    print("\n" + "=" * 60)
    print("VISUALIZATIONS GENERATED")
    print("=" * 60)
    print(f"  - {output_dir / 'ablation_study.png'}")
    print(f"  - {output_dir / 'confusion_matrix.png'}")
    print(f"  - {output_dir / 'tsne_embeddings.png'}")
    print(f"  - {output_dir / 'class_distribution.png'}")
    print(f"  - {output_dir / 'benchmark_detailed.png'}")
    print("\nDone!")


if __name__ == "__main__":
    main()
