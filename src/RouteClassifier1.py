"""
Parker Loupessis Final Project
Route Classifier class for the first model 
  Uses (x, y) coordinates of WR and snap
"""
import torch
import torch.nn as nn

class RouteClassifier1(nn.Module):
  def __init__(self, input_size=4, hidden_size=64, num_layers=2, num_classes=4, dropout=0.3):
    """
        Args:
            input_size:  number of features per timestep
            hidden_size: size of the GRU hidden state
            num_layers:  number of stacked GRU layers
            num_classes: number of route types to classify
            dropout:     dropout rate applied between GRU layers and in classifier head
      """
    super().__init__()

    # GRU processes the sequence of input across all timesteps
    # and produces a hidden state that summarizes the entire route
    # code written with AI assistance
    self.gru = nn.GRU(input_size=input_size, 
                      hidden_size=hidden_size,
                      num_layers=num_layers,
                      batch_first=True,
                      dropout=dropout if num_layers > 1 else 0
    )

    # classifier head maps the final GRU hidden state to class logits
    # code written with AI assistance
    self.pipeline = torch.nn.Sequential(
        nn.Linear(hidden_size, 128),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(64, num_classes)
    )
  
  def forward(self, x):
    # x shape: (batch, max_frames, 11) — slice first 4 features (x, y, snap_x, snap_y)
    # remaining features (relative coords, speed, direction, etc.) not used in this model
    x = x[:,:,:4]

    # pass full sequence through GRU
    out, hidden = self.gru(x)

    # take the final hidden state of the last GRU layer as the route representation
    last_hidden = hidden[-1]

    # classify the route based on the compressed hidden state
    return self.pipeline(last_hidden)