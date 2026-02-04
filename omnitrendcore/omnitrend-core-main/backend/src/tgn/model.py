"""
Advanced Temporal Graph Network (TGN) for News Trend Analysis

This module implements state-of-the-art temporal modeling techniques:

1. Time2Vec Encoding:
   - Learns both periodic patterns (daily/weekly news cycles)  
   - Captures linear time trends
   - Formula: t(τ)[0] = ω*τ + φ (linear), t(τ)[i>0] = sin(ω*τ + φ) (periodic)

2. Exponential Decay Time-Aware Aggregation:
   - Weights neighbor messages by temporal recency
   - Formula: weight = exp(-λ * time_diff) * attention
   - Recent articles have stronger influence

3. LSTM Temporal Memory:
   - Tracks topic evolution over time
   - Maintains hidden state across time windows
   - Captures long-term trend patterns
   - Gated updates balance current vs. historical information

Architecture:
    Input → Time2Vec Encoding → GAT Layers (with temporal weights) → LSTM Memory → Trend Prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
from .layers import NeuralHawkesAttention, NeuralODEMemory
import numpy as np
from typing import Dict, List, Tuple

class Time2Vec(nn.Module):
    """
    Time2Vec: Learning a Vector Representation of Time
    Captures both periodic patterns (daily/weekly cycles) and linear trends
    
    t(τ)[i] = ω_i * τ + φ_i                    if i = 0 (linear)
    t(τ)[i] = sin(ω_i * τ + φ_i)              if i > 0 (periodic)
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Learnable parameters for periodic and linear components
        self.omega = nn.Parameter(torch.randn(hidden_dim))
        self.phi = nn.Parameter(torch.randn(hidden_dim))
        
        # Additional projection layer
        self.proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, time_diffs: torch.Tensor) -> torch.Tensor:
        """
        Encode time differences using Time2Vec
        
        Args:
            time_diffs: [num_edges] or [num_edges, 1]
        
        Returns:
            time_features: [num_edges, hidden_dim]
        """
        if time_diffs.dim() == 1:
            time_diffs = time_diffs.unsqueeze(-1)  # [num_edges, 1]
        
        # Compute ω * τ + φ for all dimensions
        time_scaled = time_diffs * self.omega + self.phi  # [num_edges, hidden_dim]
        
        # First dimension is linear, rest are periodic (sin)
        time_features = torch.zeros_like(time_scaled)
        time_features[:, 0] = time_scaled[:, 0]  # Linear component
        time_features[:, 1:] = torch.sin(time_scaled[:, 1:])  # Periodic components
        
        # Project to final hidden dimension
        time_features = self.proj(time_features)
        
        return time_features


class TemporalAttentionAdvanced(nn.Module):
    """
    Advanced temporal attention with exponential decay
    Combines Time2Vec encoding with learnable attention and decay
    """
    
    def __init__(self, hidden_dim: int, decay_lambda: float = 0.0001):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.decay_lambda = nn.Parameter(torch.tensor([decay_lambda]))
        
        # Time2Vec encoder
        self.time2vec = Time2Vec(hidden_dim)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, time_diffs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute temporal attention weights with exponential decay
        
        Args:
            time_diffs: [num_edges] time differences
        
        Returns:
            attn_weights: [num_edges, 1] attention weights
            time_features: [num_edges, hidden_dim] encoded time features
        """
        # Encode time with Time2Vec
        time_features = self.time2vec(time_diffs)  # [num_edges, hidden_dim]
        
        # Compute learnable attention weights
        attn_logits = self.attention(time_features)  # [num_edges, 1]
        
        # Apply exponential decay: exp(-λ * time_diff)
        if time_diffs.dim() == 1:
            time_diffs_expanded = time_diffs.unsqueeze(-1)
        else:
            time_diffs_expanded = time_diffs
        
        decay_weights = torch.exp(-self.decay_lambda * time_diffs_expanded)
        
        # Combine learned attention with decay
        attn_weights = torch.sigmoid(attn_logits) * decay_weights
        
        # Calculate attention entropy (measure of attention focus)
        # Add numerical stability with clamp and larger epsilon
        normalized_weights = F.softmax(attn_logits.clamp(-10, 10), dim=0)
        log_weights = torch.log(normalized_weights.clamp(min=1e-6))
        entropy = -torch.sum(normalized_weights * log_weights)
        entropy = entropy.clamp(0, 100)  # Prevent extreme values
        
        return attn_weights, time_features, entropy

class TemporalMemory(nn.Module):
    """
    LSTM-based temporal memory module
    Tracks topic evolution and trend patterns over time
    """
    
    def __init__(self, hidden_dim: int, num_layers: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM for temporal dynamics
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        # Memory update gate (decides how much to rely on history)
        self.update_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
    
    def forward(
        self, 
        current_features: torch.Tensor, 
        hidden_state: Tuple[torch.Tensor, torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Update temporal memory with current features
        
        Args:
            current_features: [num_nodes, hidden_dim]
            hidden_state: (h, c) tuple from previous step
        
        Returns:
            updated_features: [num_nodes, hidden_dim]
            new_hidden_state: (h, c) tuple for next step
        """
        # Add sequence dimension: [num_nodes, 1, hidden_dim]
        current_features_seq = current_features.unsqueeze(1)
        
        # Pass through LSTM
        if hidden_state is None:
            lstm_out, (h_n, c_n) = self.lstm(current_features_seq)
        else:
            lstm_out, (h_n, c_n) = self.lstm(current_features_seq, hidden_state)
        
        # Remove sequence dimension: [num_nodes, hidden_dim]
        lstm_features = lstm_out.squeeze(1)
        
        # Gated combination of current and memory features
        combined = torch.cat([current_features, lstm_features], dim=-1)
        gate = self.update_gate(combined)
        
        updated_features = gate * lstm_features + (1 - gate) * current_features
        
        return updated_features, (h_n, c_n)


class TGNEncoder(nn.Module):
    """
    Advanced Temporal Graph Network Encoder with:
    - Time2Vec encoding for periodic patterns
    - LSTM temporal memory for trend evolution
    - Exponential decay time-aware aggregation
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_temporal_memory: bool = True,
        use_advanced_temporal: bool = False
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_temporal_memory = use_temporal_memory
        self.use_advanced_temporal = use_advanced_temporal
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        if use_advanced_temporal:
            self.temporal_attn = NeuralHawkesAttention(hidden_dim, num_heads)
            if use_temporal_memory:
                self.temporal_memory = NeuralODEMemory(hidden_dim, number_of_ode_layers=2)
        else:
            self.temporal_attn = TemporalAttentionAdvanced(hidden_dim)
            if use_temporal_memory:
                self.temporal_memory = TemporalMemory(hidden_dim, num_layers=1)
        
        # Graph attention layers
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.conv_layers.append(
                    GATv2Conv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout)
                )
            else:
                self.conv_layers.append(
                    GATv2Conv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout)
                )
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor = None,
        time_diffs: torch.Tensor = None,
        hidden_state: Tuple[torch.Tensor, torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with advanced temporal modeling
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge attributes [num_edges, edge_dim]
            time_diffs: Time differences for edges [num_edges]
            hidden_state: LSTM hidden state from previous time step (optional)
        
        Returns:
            Dictionary containing:
                - embeddings: Updated node embeddings [num_nodes, hidden_dim]
                - time_features: Encoded time features [num_edges, hidden_dim]
                - hidden_state: New LSTM hidden state (if using temporal memory)
        """
        # Project input features
        h = self.input_proj(x)
        h = F.relu(h)
        
        # Compute temporal features and attention weights
        time_features = None
        edge_weights = None
        
        if time_diffs is not None:
            # Get Time2Vec encoding and exponential decay weights
            edge_weights, time_features, entropy = self.temporal_attn(time_diffs)
            # Squeeze to [num_edges] for edge weighting
            edge_weights = edge_weights.squeeze(-1)
        
        # Apply graph convolutions with time-aware aggregation
        for i, conv in enumerate(self.conv_layers):
            h_old = h
            
            # Apply GAT convolution
            h = conv(h, edge_index)
            
            # Apply temporal weighting to first layer output
            # This weights the aggregated messages by temporal importance
            if edge_weights is not None and i == 0:
                # Compute mean temporal weight per node (aggregate from incoming edges)
                num_nodes = h.size(0)
                node_weights = torch.zeros(num_nodes, device=h.device)
                node_counts = torch.zeros(num_nodes, device=h.device)
                
                # Aggregate edge weights to target nodes
                target_nodes = edge_index[1]  # Target nodes
                node_weights.index_add_(0, target_nodes, edge_weights)
                node_counts.index_add_(0, target_nodes, torch.ones_like(edge_weights))
                
                # Average weights per node (avoid division by zero)
                node_counts = torch.clamp(node_counts, min=1.0)
                node_weights = node_weights / node_counts
                
                # Apply temporal weighting
                h = h * node_weights.unsqueeze(-1)
            
            # Residual connection (skip connection)
            if i > 0:
                h = h + h_old
            
            # Layer norm and activation
            h = self.layer_norms[i](h)
            h = F.relu(h)
            h = self.dropout(h)
        
        # Apply LSTM temporal memory to capture evolution over time
        new_hidden_state = None
        if self.use_temporal_memory:
            h, new_hidden_state = self.temporal_memory(h, hidden_state)
        
        result = {
            'embeddings': h,
            'time_features': time_features
        }
        
        if 'entropy' in locals():
            result['attention_entropy'] = entropy
        
        if new_hidden_state is not None:
            result['hidden_state'] = new_hidden_state
        
        return result

class TrendPredictor(nn.Module):
    """Predicts trend strength for news articles"""
    
    def __init__(self, hidden_dim: int, num_trend_classes: int = 3):
        super().__init__()
        
        self.trend_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_trend_classes)
        )
        
        self.trend_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, node_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict trend class and score
        
        Returns:
            trend_class: [num_nodes, num_classes] - probability distribution
            trend_score: [num_nodes, 1] - continuous trend strength
        """
        trend_class = F.softmax(self.trend_classifier(node_embeddings), dim=-1)
        trend_score = self.trend_scorer(node_embeddings)
        
        return trend_class, trend_score

class NewsTGN(nn.Module):
    """Complete Temporal Graph Network for news trend analysis"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 4,
        num_trend_classes: int = 3,
        dropout: float = 0.1,
        use_advanced_temporal: bool = False
    ):
        super().__init__()
        
        self.encoder = TGNEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            use_advanced_temporal=use_advanced_temporal
        )
        
        self.trend_predictor = TrendPredictor(
            hidden_dim=hidden_dim,
            num_trend_classes=num_trend_classes
        )
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor = None,
        time_diffs: torch.Tensor = None,
        hidden_state: Tuple[torch.Tensor, torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with advanced temporal modeling
        
        Returns:
            Dictionary containing:
                - embeddings: Node embeddings
                - trend_class: Trend classification
                - trend_score: Trend strength scores
                - time_features: Encoded time features (if time_diffs provided)
                - hidden_state: LSTM hidden state (if using temporal memory)
        """
        # Encode graph with advanced temporal features
        encoder_output = self.encoder(x, edge_index, edge_attr, time_diffs, hidden_state)
        
        # Extract embeddings
        embeddings = encoder_output['embeddings']
        
        # Predict trends
        trend_class, trend_score = self.trend_predictor(embeddings)
        
        # Build output dictionary
        output = {
            'embeddings': embeddings,
            'trend_class': trend_class,
            'trend_score': trend_score
        }
        
        # Add time features if available
        if 'time_features' in encoder_output and encoder_output['time_features'] is not None:
            output['time_features'] = encoder_output['time_features']
        
        # Add hidden state if available
        if 'hidden_state' in encoder_output:
            output['hidden_state'] = encoder_output['hidden_state']
            
        # Add attention entropy if available
        if 'attention_entropy' in encoder_output:
            output['attention_entropy'] = encoder_output['attention_entropy']
        
        return output
    
    def compute_graph_embedding(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute global graph embedding via mean pooling"""
        return node_embeddings.mean(dim=0)
    
    def detect_trending_topics(
        self,
        node_embeddings: torch.Tensor,
        trend_scores: torch.Tensor,
        node_topics: List[List[str]],
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Detect trending topics based on node importance and trend scores
        
        Returns:
            List of (topic, score) tuples
        """
        # Aggregate scores by topic
        topic_scores = {}
        
        for i, topics in enumerate(node_topics):
            score = trend_scores[i].item()
            for topic in topics:
                if topic not in topic_scores:
                    topic_scores[topic] = []
                topic_scores[topic].append(score)
        
        # Compute average score per topic
        topic_avg_scores = {
            topic: np.mean(scores)
            for topic, scores in topic_scores.items()
        }
        
        # Sort and return top-k
        trending = sorted(
            topic_avg_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        return trending

# Example usage
if __name__ == "__main__":
    # Create sample data
    num_nodes = 100
    input_dim = 385  # 384 from sentence-transformers + 1 sentiment
    
    # Random features
    x = torch.randn(num_nodes, input_dim)
    
    # Random edges
    num_edges = 300
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # Random time differences
    time_diffs = torch.rand(num_edges) * 86400  # Random times within a day
    
    # Create model
    model = NewsTGN(
        input_dim=input_dim,
        hidden_dim=256,
        num_layers=3,
        num_trend_classes=3
    )
    
    # Forward pass
    output = model(x, edge_index, time_diffs=time_diffs)
    
    print(f"Embeddings shape: {output['embeddings'].shape}")
    print(f"Trend class shape: {output['trend_class'].shape}")
    print(f"Trend score shape: {output['trend_score'].shape}")
    
    # Detect trending topics (example)
    node_topics = [['economy', 'politics'] for _ in range(num_nodes)]
    trending = model.detect_trending_topics(
        output['embeddings'],
        output['trend_score'],
        node_topics,
        top_k=5
    )
    print(f"\nTop trending topics: {trending}")
