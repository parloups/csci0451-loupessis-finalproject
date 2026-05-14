"""
Parker Loupessis Final Project
Route Classifier class for the first model 
  Uses (x, y) coordinates of WR and snap
"""
import torch
import torch.nn as nn

class RouteClassifier1(nn.Module):
  def __init__(self, input_size=4, hidden_size=64, num_layers=2, num_classes=4, dropout=0.3):
    super().__init__()

    self.gru = nn.GRU(input_size=input_size, 
                      hidden_size=hidden_size,
                      num_layers=num_layers,
                      batch_first=True,
                      dropout=dropout if num_layers > 1 else 0
    )
    
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
    x = x[:,:,:4]
    out, hidden = self.gru(x)
    last_hidden = hidden[-1]
    return self.pipeline(last_hidden)