"""
Baseline Models for Comparison

This module implements baseline methods for comparing against the TGN approach:
1. TF-IDF + Cosine Similarity (no deep learning)
2. MLP on Embeddings (no graph structure)
3. Static GCN (no temporal features)
4. Standard TGN (without Time2Vec/Hawkes enhancements)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple


class TFIDFBaseline:
    """TF-IDF + Cosine Similarity baseline (no deep learning)"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000)
        
    def fit_predict(self, texts: List[str], labels: np.ndarray) -> np.ndarray:
        """
        Predict trend scores based on TF-IDF centrality
        Articles similar to many others get higher scores
        """
        # Compute TF-IDF vectors
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        
        # Compute pairwise cosine similarities
        sim_matrix = cosine_similarity(tfidf_matrix)
        
        # Score = average similarity to other articles (centrality)
        scores = sim_matrix.mean(axis=1)
        
        # Normalize to [0, 1]
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        
        # Convert to class predictions (0, 1, 2)
        predictions = np.zeros(len(scores), dtype=np.int64)
        predictions[scores >= 0.33] = 1
        predictions[scores >= 0.66] = 2
        
        return predictions, scores


class MLPBaseline(nn.Module):
    """MLP on embeddings (ignores graph structure)"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_classes: int = 3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Get intermediate features
        h = self.mlp[:-1](x)  # All but last layer
        
        # Classification
        logits = self.mlp[-1](h)
        trend_class = F.softmax(logits, dim=-1)
        
        # Regression
        trend_score = self.score_head(h)
        
        return {
            'trend_class': trend_class,
            'trend_score': trend_score
        }


class StaticGCNBaseline(nn.Module):
    """Static GCN (no temporal features)"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_classes: int = 3, num_layers: int = 2):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        # Ignore any temporal features passed in kwargs
        h = self.input_proj(x)
        h = F.relu(h)
        
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=0.1, training=self.training)
        
        trend_class = F.softmax(self.classifier(h), dim=-1)
        trend_score = self.scorer(h)
        
        return {
            'trend_class': trend_class,
            'trend_score': trend_score
        }


class StandardTGNBaseline(nn.Module):
    """
    Standard TGN without advanced temporal features
    (No Time2Vec, No Neural Hawkes, No Neural ODE)
    Just basic time-aware attention
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_classes: int = 3, num_layers: int = 2, num_heads: int = 4):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Simple time encoding (just linear projection)
        self.time_encoder = nn.Linear(1, hidden_dim)
        
        # GAT layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads))
        
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, time_diffs: torch.Tensor = None, **kwargs) -> Dict[str, torch.Tensor]:
        h = self.input_proj(x)
        h = F.relu(h)
        
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=0.1, training=self.training)
        
        trend_class = F.softmax(self.classifier(h), dim=-1)
        trend_score = self.scorer(h)
        
        return {
            'trend_class': trend_class,
            'trend_score': trend_score
        }


class RecencyBaseline:
    """Simple recency-based baseline (newer = higher trend)"""
    
    def predict(self, timestamps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Normalize timestamps to [0, 1]
        scores = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min() + 1e-8)
        
        # Convert to class predictions
        predictions = np.zeros(len(scores), dtype=np.int64)
        predictions[scores >= 0.33] = 1
        predictions[scores >= 0.66] = 2
        
        return predictions, scores


class DegreeBaseline:
    """Degree centrality baseline (more connections = higher trend)"""
    
    def predict(self, edge_index: torch.Tensor, num_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
        # Compute in-degree for each node
        degrees = torch.zeros(num_nodes)
        for i in range(edge_index.shape[1]):
            degrees[edge_index[1, i]] += 1
        
        degrees = degrees.numpy()
        
        # Normalize
        scores = (degrees - degrees.min()) / (degrees.max() - degrees.min() + 1e-8)
        
        # Convert to class predictions
        predictions = np.zeros(len(scores), dtype=np.int64)
        predictions[scores >= 0.33] = 1
        predictions[scores >= 0.66] = 2
        
        return predictions, scores


class LSTMBaseline(nn.Module):
    """LSTM on embeddings (sequential model, no graph structure)"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_classes: int = 3, num_layers: int = 2):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        # x: [num_nodes, input_dim] -> treat as sequence
        # Add batch dimension if needed
        if len(x.shape) == 2:
            x = x.unsqueeze(0)  # [1, num_nodes, input_dim]
        
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use output at each position
        h = lstm_out.squeeze(0)  # [num_nodes, hidden_dim]
        
        trend_class = F.softmax(self.classifier(h), dim=-1)
        trend_score = self.scorer(h)
        
        return {
            'trend_class': trend_class,
            'trend_score': trend_score
        }


class GATBaseline(nn.Module):
    """Graph Attention Network baseline (graph structure, no temporal)"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_classes: int = 3, 
                 num_layers: int = 2, num_heads: int = 4):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GAT layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_channels = hidden_dim
            out_channels = hidden_dim // num_heads
            self.convs.append(GATConv(in_channels, out_channels, heads=num_heads, dropout=0.1))
        
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        h = self.input_proj(x)
        h = F.relu(h)
        
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.elu(h)
            h = F.dropout(h, p=0.1, training=self.training)
        
        trend_class = F.softmax(self.classifier(h), dim=-1)
        trend_score = self.scorer(h)
        
        return {
            'trend_class': trend_class,
            'trend_score': trend_score
        }


class SVMBaseline:
    """SVM baseline using sklearn (traditional ML)"""
    
    def __init__(self):
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        self.model = SVC(kernel='rbf', probability=True, class_weight='balanced')
        
    def fit_predict(self, features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        from sklearn.model_selection import train_test_split
        
        # Scale features
        X_scaled = self.scaler.fit_transform(features)
        
        # Fit model
        self.model.fit(X_scaled, labels)
        
        # Predict
        predictions = self.model.predict(X_scaled)
        proba = self.model.predict_proba(X_scaled)
        
        # Score = probability of being in highest class
        scores = proba[:, -1] if proba.shape[1] > 1 else proba[:, 0]
        
        return predictions, scores


class RandomForestBaseline:
    """Random Forest baseline using sklearn"""
    
    def __init__(self, n_estimators: int = 100):
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=10,
            class_weight='balanced',
            random_state=42
        )
        
    def fit_predict(self, features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Fit model
        self.model.fit(features, labels)
        
        # Predict
        predictions = self.model.predict(features)
        proba = self.model.predict_proba(features)
        
        # Score = probability of being in highest class
        scores = proba[:, -1] if proba.shape[1] > 1 else proba[:, 0]
        
        return predictions, scores

