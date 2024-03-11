import torch
import torch.nn as nn
import torch.nn.functional as F
from attention_mechanism import AttentionMechanism

class GraphAttentionLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout_rate: float, is_output_layer: bool = False):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.is_output_layer = is_output_layer
        
        self.weight_matrix = nn.Parameter(torch.empty(input_dim, output_dim))
        nn.init.orthogonal_(self.weight_matrix)
        
        self.attention_mechanism = AttentionMechanism(output_dim)

    def forward(self, node_features: torch.Tensor, adjacency_matrix: torch.Tensor) -> torch.Tensor:
        transformed_features = torch.mm(node_features, self.weight_matrix)
        attention_coefficients = self.attention_mechanism(transformed_features)
        
        mask = -1e15 * torch.ones_like(attention_coefficients)
        masked_attention = torch.where(adjacency_matrix > 0, attention_coefficients, mask)
        attention_weights = F.softmax(masked_attention, dim=1)
        attention_weights = F.dropout(attention_weights, self.dropout_rate, training=self.training)
        
        node_outputs = torch.matmul(attention_weights, transformed_features)
        
        return node_outputs if self.is_output_layer else F.elu(node_outputs)