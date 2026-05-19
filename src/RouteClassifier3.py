"""
Parker Loupessis Final Project
Route Classifier class for the third model
  Uses (x, y) coordinates relative to the snap of WR
  Encodes (x, y) coordinates of the snap seperately
  Uses speed, direction, and distance traveled
  Uses randomly augmented data for training
  Ignores padding
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class RouteClassifier3(nn.Module):
  def __init__(self, input_size=6, snap_size=2, hidden_size=64, num_layers=2, num_classes=4, dropout=0.3):
    """
        Trained on augmented data to address class imbalance.

        Args:
            input_size:  number of sequence features per timestep
            snap_size:   number of snap context features
            hidden_size: size of the GRU hidden state
            num_layers:  number of stacked GRU layers
            num_classes: number of route types to classify
            dropout:     dropout rate applied between GRU layers and in classifier head
      """
    super().__init__()

    # encodes the snap location (x, y)
    # code written with AI assistance
    self.snap_encoder = nn.Sequential(
        nn.Linear(snap_size, 16),
        nn.ReLU()
    )

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
        nn.Linear(hidden_size + 16, 128),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(64, num_classes)
    )
  
  def forward(self, x):
    # extract real route lengths from the last feature column
    # used by pack_padded_sequence to ignore zero-padded timesteps
    lengths = x[:, 0, -1].long()

    # extract snap location from first timestep
    snap_location = x[:,0,2:4]

    # select sequence features: relative_x(4), relative_y(5), s(6), dir_sin(7), dir_cos(8), dis(9)
    x = x[:,:, [4,5,6,7,8,9]]

    # encode snap location into context vector
    snap_encoded = self.snap_encoder(snap_location)

    # pack sequence to skip padded timesteps during GRU forward pass
    # code written with AI assistance
    packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
    out, hidden = self.gru(packed)
    out, _ = pad_packed_sequence(out, batch_first=True)

    # take the final hidden state of the last GRU layer as the route representation
    last_hidden = hidden[-1]

    # concatenate route representation with snap context
    combined = torch.cat([last_hidden, snap_encoded], dim=1)
    return self.pipeline(combined)
