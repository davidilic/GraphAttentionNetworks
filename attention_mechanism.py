import torch
import torch.nn as nn

class AttentionMechanism(nn.Module):
    def __init__(self, feature_dim: int, alpha: float = 0.2):
        super().__init__()
        self.attention_vector = nn.Parameter(torch.empty(2 * feature_dim, 1))
        nn.init.kaiming_uniform_(self.attention_vector)
        self.leaky_relu = nn.LeakyReLU(alpha)

    def forward(self, feature_matrix: torch.Tensor) -> torch.Tensor:
        N = feature_matrix.size(0)
        combined_features = torch.cat([feature_matrix.repeat(1, N).view(N * N, -1),
                                       feature_matrix.repeat(N, 1)], dim=1).view(N, -1, 2 * feature_matrix.size(1))
        e = self.leaky_relu(torch.matmul(combined_features, self.attention_vector).squeeze(2))
        return e