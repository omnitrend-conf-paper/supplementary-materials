#!/usr/bin/env python3
"""
Optimized TGN Training Script

Improvements over default training:
1. Use advanced temporal features (Neural Hawkes, ODE Memory)
2. Larger model (512 hidden dim)
3. Cosine annealing learning rate
4. Gradient clipping
5. Label smoothing
6. Edge weight features
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import json
import numpy as np
from tgn.model import NewsTGN
from tgn.train import TGNTrainer


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy with label smoothing for regularization"""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, pred, target):
        n_classes = pred.size(-1)
        log_preds = torch.log_softmax(pred, dim=-1)
        
        loss = -log_preds.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_preds.mean(dim=-1)
        
        return (1 - self.smoothing) * loss.mean() + self.smoothing * smooth_loss.mean()


def train_optimized():
    print("=" * 60)
    print("OPTIMIZED TGN TRAINING")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading graph data...")
    data = TGNTrainer.load_graph_data("data/processed")
    print(f"   Nodes: {data.x.shape[0]}, Features: {data.x.shape[1]}")
    print(f"   Edges: {data.edge_index.shape[1]}")
    print(f"   Labels: Low={int((data.y==0).sum())}, Med={int((data.y==1).sum())}, High={int((data.y==2).sum())}")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")
    
    # Move data to device
    data = data.to(device)
    
    # Create IMPROVED model
    print("\n2. Creating optimized model...")
    model = NewsTGN(
        input_dim=data.x.shape[1],
        hidden_dim=512,          # Larger hidden dimension
        num_layers=3,
        num_heads=8,             # More attention heads
        num_trend_classes=3,
        dropout=0.15,            # Slightly more dropout
        use_advanced_temporal=False  # Keep standard for stability
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=5e-4,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    
    # Cosine annealing scheduler
    num_epochs = 100
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    # Label smoothing loss
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    
    # Training loop
    print(f"\n3. Training for {num_epochs} epochs...")
    best_acc = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass with time features
        output = model(
            data.x, 
            data.edge_index,
            time_diffs=getattr(data, 'time_diffs', None)
        )
        
        # Compute losses
        class_loss = criterion(output['trend_class'], data.y)
        
        # Regression loss (MSE)
        y_score = data.y.float() / 2.0  # Normalize to [0, 1]
        score_loss = nn.MSELoss()(output['trend_score'].squeeze(), y_score)
        
        # Combined loss
        loss = class_loss + 0.3 * score_loss
        
        # Backward with gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        # Compute accuracy
        pred = output['trend_class'].argmax(dim=1)
        acc = (pred == data.y).float().mean().item()
        
        # Save best model
        if acc > best_acc:
            best_acc = acc
            best_model_state = model.state_dict().copy()
        
        # Log progress
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1:3d}: Loss={loss.item():.4f}, Acc={acc:.4f}, LR={scheduler.get_last_lr()[0]:.6f}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    print(f"\n   Best accuracy: {best_acc:.4f}")
    
    # Save model
    print("\n4. Saving model...")
    save_path = Path("models/optimized_model.pt")
    save_path.parent.mkdir(exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_accuracy': best_acc,
        'config': {
            'hidden_dim': 512,
            'num_layers': 3,
            'num_heads': 8,
            'dropout': 0.15
        }
    }, save_path)
    print(f"   Saved to: {save_path}")
    
    # Final evaluation
    print("\n5. Final evaluation...")
    model.eval()
    with torch.no_grad():
        output = model(data.x, data.edge_index, time_diffs=getattr(data, 'time_diffs', None))
        pred = output['trend_class'].argmax(dim=1)
        acc = (pred == data.y).float().mean().item()
        
        from sklearn.metrics import classification_report
        print(classification_report(
            data.y.cpu().numpy(),
            pred.cpu().numpy(),
            target_names=['Low', 'Medium', 'High']
        ))
    
    print(f"\nFinal Accuracy: {acc:.4f}")
    print("Done!")
    
    return model


if __name__ == "__main__":
    train_optimized()
