"""Simple MLP regressor for sequence inputs."""
import torch
import torch.nn as nn


class MLPRegressor(nn.Module):
    """
    Simple feedforward MLP for regression on sequence data.
    Uses average pooling over time dimension before feeding to MLP.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.1):
        """
        Args:
            input_dim: Feature dimension (F)
            hidden_dim: Hidden layer size
            dropout: Dropout probability
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, T, F)
        
        Returns:
            Predictions of shape (B,)
        """
        # Average pool over time dimension
        x = x.mean(dim=1)  # (B, T, F) -> (B, F)
        
        # Pass through MLP
        out = self.net(x)  # (B, 1)
        
        return out.squeeze(1)  # (B,)
