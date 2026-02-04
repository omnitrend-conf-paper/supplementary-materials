import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class NeuralHawkesAttention(nn.Module):
    def __init__(self, hidden_dimension: int, number_of_attention_heads: int = 4):
        super().__init__()
        self.hidden_dimension = hidden_dimension
        self.number_of_attention_heads = number_of_attention_heads
        
        self.base_intensity_mu = nn.Parameter(torch.ones(1) * 0.1)
        self.excitation_strength_alpha = nn.Parameter(torch.ones(number_of_attention_heads, 1) * 0.5)
        self.decay_rate_beta = nn.Parameter(torch.ones(number_of_attention_heads, 1) * 1.0)
        
        self.intensity_projection_network = nn.Sequential(
            nn.Linear(number_of_attention_heads, hidden_dimension),
            nn.ReLU(),
            nn.Linear(hidden_dimension, hidden_dimension)
        )
    
    def forward(self, time_differences: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size_edges = time_differences.size(0)
        time_values_expanded = time_differences.view(-1, 1, 1)
        
        decay_exponential = torch.exp(-torch.abs(self.decay_rate_beta).unsqueeze(0) * time_values_expanded)
        excitation_term = torch.abs(self.excitation_strength_alpha).unsqueeze(0) * decay_exponential
        
        intensity_by_head = self.base_intensity_mu + excitation_term
        intensity_averaged = intensity_by_head.mean(dim=1).squeeze(-1)
        
        edge_scalar_weights = torch.sigmoid(intensity_averaged).unsqueeze(-1)
        
        hawkes_features_projection = self.intensity_projection_network(intensity_by_head.squeeze(-1))
        
        return edge_scalar_weights, hawkes_features_projection


class NeuralODEMemory(nn.Module):
    def __init__(self, hidden_dimension: int, number_of_ode_layers: int = 2):
        super().__init__()
        self.hidden_dimension = hidden_dimension
        
        ode_network_layers = []
        for layer_index in range(number_of_ode_layers):
            if layer_index == 0:
                ode_network_layers.append(nn.Linear(hidden_dimension + 1, hidden_dimension))
            else:
                ode_network_layers.append(nn.Linear(hidden_dimension, hidden_dimension))
            ode_network_layers.append(nn.Tanh())
        
        self.dynamics_function_network = nn.Sequential(*ode_network_layers)
        
        self.state_update_gate = nn.Linear(hidden_dimension * 2, hidden_dimension)
    
    def compute_derivative(self, current_hidden_state: torch.Tensor, current_time: torch.Tensor) -> torch.Tensor:
        time_feature = current_time.expand(current_hidden_state.size(0), 1)
        state_with_time = torch.cat([current_hidden_state, time_feature], dim=-1)
        derivative = self.dynamics_function_network(state_with_time)
        return derivative
    
    def euler_integration_step(
        self, 
        initial_state: torch.Tensor, 
        start_time: float, 
        end_time: float, 
        number_of_steps: int = 10
    ) -> torch.Tensor:
        time_step_size = (end_time - start_time) / number_of_steps
        current_state = initial_state
        current_time_value = start_time
        
        for step_index in range(number_of_steps):
            time_tensor = torch.tensor([[current_time_value]], device=initial_state.device)
            state_derivative = self.compute_derivative(current_state, time_tensor)
            current_state = current_state + time_step_size * state_derivative
            current_time_value += time_step_size
        
        return current_state
    
    def forward(
        self, 
        current_node_features: torch.Tensor, 
        previous_hidden_state: Optional[torch.Tensor] = None,
        time_interval: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if previous_hidden_state is None:
            previous_hidden_state = torch.zeros_like(current_node_features)
        
        evolved_state = self.euler_integration_step(
            previous_hidden_state, 
            start_time=0.0, 
            end_time=time_interval,
            number_of_steps=10
        )
        
        combined_features = torch.cat([current_node_features, evolved_state], dim=-1)
        update_gate_values = torch.sigmoid(self.state_update_gate(combined_features))
        
        updated_hidden_state = update_gate_values * evolved_state + (1 - update_gate_values) * current_node_features
        
        return updated_hidden_state, updated_hidden_state
