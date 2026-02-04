import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from datetime import datetime
import gc
from tqdm import tqdm

from .model import NewsTGN

class TGNTrainer:
    """Trainer for Temporal Graph Network on news data"""
    
    def __init__(
        self,
        model: NewsTGN,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        device: str = None
    ):
        self.model = model
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Loss history
        self.train_losses = []
        self.val_losses = []
        
    def load_graph_data(self, data_dir: str = "data/processed") -> Data:
        """Load graph data from saved files"""
        data_dir = Path(data_dir)
        
        # Load node features
        with open(data_dir / "node_features.json", 'r') as f:
            node_feat_dict = json.load(f)
        
        # Convert to tensor - handle both string and int keys
        node_ids = sorted(node_feat_dict.keys(), key=str)
        node_id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
        
        node_features = torch.tensor([
            node_feat_dict[nid] for nid in node_ids
        ], dtype=torch.float32)
        
        # Load graph structure
        with open(data_dir / "graph_structure.json", 'r') as f:
            graph_data = json.load(f)
        
        # Build edge index (NetworkX 3.x uses 'edges', 2.x uses 'links')
        edge_list = graph_data.get('links') or graph_data.get('edges', [])
        edges = []
        for link in edge_list:
            src_id = str(link['source']) if not isinstance(link['source'], dict) else str(link['source'].get('id', link['source']))
            tgt_id = str(link['target']) if not isinstance(link['target'], dict) else str(link['target'].get('id', link['target']))
            
            if src_id in node_id_to_idx and tgt_id in node_id_to_idx:
                src = node_id_to_idx[src_id]
                tgt = node_id_to_idx[tgt_id]
                edges.append([src, tgt])
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        # Load timestamps
        with open(data_dir / "timestamps.json", 'r') as f:
            timestamps_dict = json.load(f)
        
        # Compute time differences for edges
        time_diffs = []
        for link in edge_list:
            src_id = str(link['source']) if not isinstance(link['source'], dict) else str(link['source'].get('id', link['source']))
            tgt_id = str(link['target']) if not isinstance(link['target'], dict) else str(link['target'].get('id', link['target']))
            
            src_time = timestamps_dict.get(src_id, 0)
            tgt_time = timestamps_dict.get(tgt_id, 0)
            time_diffs.append(abs(tgt_time - src_time))
        
        time_diffs = torch.tensor(time_diffs, dtype=torch.float32)
        
        # Normalize time_diffs to [0, 1] to prevent numerical instability
        if len(time_diffs) > 0 and time_diffs.max() > 0:
            time_diffs = time_diffs / time_diffs.max()
        
        # Normalize node features (standardize)
        node_features = (node_features - node_features.mean(dim=0)) / (node_features.std(dim=0) + 1e-8)
        
        # Replace any NaN/Inf with 0
        node_features = torch.nan_to_num(node_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Create PyG Data object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            time_diffs=time_diffs
        )
        
        # Generate pseudo-labels for self-supervised learning
        # Label 0: Low trend, 1: Medium trend, 2: High trend
        # Based on node degree and time recency
        degrees = torch.zeros(len(node_ids))
        for i in range(edge_index.shape[1]):
            degrees[edge_index[1, i]] += 1  # In-degree
        
        # Normalize degrees
        max_degree = degrees.max()
        if max_degree > 0:
            norm_degrees = degrees / max_degree
        else:
            norm_degrees = degrees
        
        # Create labels based on degree percentiles for better class balance
        labels = torch.zeros(len(node_ids), dtype=torch.long)
        threshold_low = torch.quantile(norm_degrees, 0.5)  # Bottom 50% = Low
        threshold_high = torch.quantile(norm_degrees, 0.85)  # Top 15% = High
        labels[norm_degrees < threshold_low] = 0
        labels[(norm_degrees >= threshold_low) & (norm_degrees < threshold_high)] = 1
        labels[norm_degrees >= threshold_high] = 2
        
        data.y = labels
        
        return data
    
    def train_epoch(self, loader: NeighborLoader) -> float:
        """Train for one epoch using mini-batches"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in tqdm(loader, desc="Training", leave=False):
            self.optimizer.zero_grad()
            
            # Move batch to device
            batch = batch.to(self.device)
            
            # Forward pass
            # Note: NeighborLoader re-indexes nodes, so we use batch.x and batch.edge_index
            # batch.time_diffs is automatically sliced by NeighborLoader
            output = self.model(
                batch.x,
                batch.edge_index,
                time_diffs=batch.time_diffs if hasattr(batch, 'time_diffs') else None
            )
            
            # Compute loss (classification + regression)
            # Only compute loss on the target nodes (first batch_size nodes)
            # NeighborLoader puts target nodes first
            batch_size = batch.batch_size
            target_y = batch.y[:batch_size]
            pred_class = output['trend_class'][:batch_size]
            pred_score = output['trend_score'][:batch_size]
            
            class_loss = nn.CrossEntropyLoss()(pred_class, target_y)
            
            # Regression loss (trend score should correlate with class)
            target_scores = target_y.float() / 2.0  # Normalize to [0, 1]
            score_loss = nn.MSELoss()(pred_score.squeeze(), target_scores)
            
            # Combined loss
            loss = class_loss + 0.5 * score_loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Clear cache to prevent OOM
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        return total_loss / num_batches if num_batches > 0 else 0
    
    def validate(self, loader: NeighborLoader) -> Tuple[float, Dict]:
        """Validate model using mini-batches"""
        self.model.eval()
        total_loss = 0
        total_acc = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Validating", leave=False):
                batch = batch.to(self.device)
                
                output = self.model(
                    batch.x,
                    batch.edge_index,
                    time_diffs=batch.time_diffs if hasattr(batch, 'time_diffs') else None
                )
                
                # Only evaluate on target nodes
                batch_size = batch.batch_size
                target_y = batch.y[:batch_size]
                pred_class = output['trend_class'][:batch_size]
                pred_score = output['trend_score'][:batch_size]
                
                # Compute losses
                class_loss = nn.CrossEntropyLoss()(pred_class, target_y)
                
                target_scores = target_y.float() / 2.0
                score_loss = nn.MSELoss()(pred_score.squeeze(), target_scores)
                
                loss = class_loss + 0.5 * score_loss
                
                # Compute accuracy
                pred_classes = pred_class.argmax(dim=1)
                accuracy = (pred_classes == target_y).float().mean()
                
                total_loss += loss.item()
                total_acc += accuracy.item()
                num_batches += 1
                
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            avg_acc = total_acc / num_batches if num_batches > 0 else 0
            
            metrics = {
                'loss': avg_loss,
                'accuracy': avg_acc
            }
        
        return avg_loss, metrics
    
    def train(
        self,
        data: Data,
        num_epochs: int = 100,
        val_data: Data = None,
        save_dir: str = "models",
        batch_size: int = 2048,
        num_neighbors: List[int] = [10, 10]
    ):
        """Train the model with mini-batching"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        best_val_loss = float('inf')
        
        print(f"Training on device: {self.device}")
        print(f"Number of nodes: {data.x.shape[0]}")
        print(f"Number of edges: {data.edge_index.shape[1]}")
        print(f"Batch size: {batch_size}")
        print(f"Neighbor sampling: {num_neighbors}")
        
        # Create loaders
        # For training, we iterate over all nodes
        train_loader = NeighborLoader(
            data,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2 if self.device.type == 'cpu' else 0
        )
        
        # For validation, we also use NeighborLoader
        if val_data is None:
            val_data = data
            
        val_loader = NeighborLoader(
            val_data,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2 if self.device.type == 'cpu' else 0
        )
        
        for epoch in range(num_epochs):
            try:
                # Train
                train_loss = self.train_epoch(train_loader)
                self.train_losses.append(train_loss)
                
                # Validate
                val_loss, val_metrics = self.validate(val_loader)
                self.val_losses.append(val_loss)
                
                # Update learning rate
                self.scheduler.step(val_loss)
                
                # Print progress
                if (epoch + 1) % 1 == 0:  # Print every epoch since they take longer now
                    print(f"Epoch {epoch + 1}/{num_epochs}")
                    print(f"  Train Loss: {train_loss:.4f}")
                    print(f"  Val Loss: {val_loss:.4f}")
                    print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(save_dir / "best_model.pt")
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"WARNING: CUDA OOM at epoch {epoch}. Clearing cache and trying to continue...")
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                        gc.collect()
                else:
                    raise e
        
        # Save final model
        self.save_checkpoint(save_dir / "final_model.pt")
        
        # Plot training curves
        self.plot_training_curves(save_dir)
        
        print(f"\nTraining completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")
    
    def save_checkpoint(self, path: Path):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, path)
    
    def load_checkpoint(self, path: Path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
    
    def plot_training_curves(self, save_dir: Path):
        """Plot and save training curves"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Curves')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
        plt.close()

# Example usage
if __name__ == "__main__":
    # Load data
    trainer_obj = TGNTrainer(
        model=NewsTGN(input_dim=385, hidden_dim=256),
        learning_rate=1e-3
    )
    
    data = trainer_obj.load_graph_data("data/processed")
    
    # Train model
    trainer_obj.train(
        data=data,
        num_epochs=100,
        save_dir="models"
    )
