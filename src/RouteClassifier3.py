"""
Parker Loupessis Final Project
Route Classifier class for the second model
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
    super().__init__()

    self.snap_encoder = nn.Sequential(
        nn.Linear(snap_size, 16),
        nn.ReLU()
    )

    self.gru = nn.GRU(input_size=input_size, 
                      hidden_size=hidden_size,
                      num_layers=num_layers,
                      batch_first=True,
                      dropout=dropout if num_layers > 1 else 0
    )
    
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
    lengths = x[:, 0, -1].long()
    snap_location = x[:,0,2:4]
    x = x[:,:, [4,5,6,7,8,9]]

    snap_encoded = self.snap_encoder(snap_location)

    packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
    out, hidden = self.gru(packed)
    out, _ = pad_packed_sequence(out, batch_first=True)
    last_hidden = hidden[-1]

    combined = torch.cat([last_hidden, snap_encoded], dim=1)
    return self.pipeline(combined)
