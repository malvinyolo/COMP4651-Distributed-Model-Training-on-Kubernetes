"""
Models: LSTM and GRU sequence classifiers for binary classification
"""
import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """
    LSTM-based binary classifier.
    Outputs logits (apply sigmoid externally).
    """
    
    def __init__(self, input_dim: int, hidden: int = 64, layers: int = 1, dropout: float = 0.1):
        """
        Args:
            input_dim: Number of input features
            hidden: Hidden layer size
            layers: Number of LSTM layers
            dropout: Dropout rate (set to 0 if layers==1 to avoid warning)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden = hidden
        self.layers = layers
        
        # If only 1 layer, set dropout to 0 to avoid PyTorch warning
        lstm_dropout = 0.0 if layers == 1 else dropout
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=layers,
            dropout=lstm_dropout,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 1)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, T, F)
        
        Returns:
            logits: Tensor of shape (B,)
        """
        # LSTM output: (B, T, hidden)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Take last hidden state: (B, hidden)
        last_hidden = lstm_out[:, -1, :]
        
        # Apply dropout and linear layer
        out = self.dropout(last_hidden)
        logits = self.fc(out).squeeze(-1)  # (B,)
        
        return logits


class GRUClassifier(nn.Module):
    """
    GRU-based binary classifier.
    Outputs logits (apply sigmoid externally).
    """
    
    def __init__(self, input_dim: int, hidden: int = 64, layers: int = 1, dropout: float = 0.1):
        """
        Args:
            input_dim: Number of input features
            hidden: Hidden layer size
            layers: Number of GRU layers
            dropout: Dropout rate (set to 0 if layers==1 to avoid warning)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden = hidden
        self.layers = layers
        
        # If only 1 layer, set dropout to 0 to avoid PyTorch warning
        gru_dropout = 0.0 if layers == 1 else dropout
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=layers,
            dropout=gru_dropout,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 1)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, T, F)
        
        Returns:
            logits: Tensor of shape (B,)
        """
        # GRU output: (B, T, hidden)
        gru_out, h_n = self.gru(x)
        
        # Take last hidden state: (B, hidden)
        last_hidden = gru_out[:, -1, :]
        
        # Apply dropout and linear layer
        out = self.dropout(last_hidden)
        logits = self.fc(out).squeeze(-1)  # (B,)
        
        return logits
