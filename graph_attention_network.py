import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from graph_attention_layer import GraphAttentionLayer

class GraphAttentionNetwork(nn.Module):
    def __init__(self, input_features: int, hidden_units: int, num_classes: int, dropout_rate: float = 0.6):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.input_layer = GraphAttentionLayer(input_features, hidden_units, dropout_rate)
        self.output_layer = GraphAttentionLayer(hidden_units, num_classes, dropout_rate, is_output_layer=True)

    def forward(self, data: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        node_features, edge_index = data
        adjacency_matrix = torch.zeros(node_features.size(0), node_features.size(0), device=node_features.device)
        adjacency_matrix[edge_index[0], edge_index[1]] = 1
        
        x = F.dropout(node_features, p=self.dropout_rate, training=self.training)
        x = self.input_layer(x, adjacency_matrix)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.output_layer(x, adjacency_matrix)
        
        return F.log_softmax(x, dim=1)