"""
Parker Loupessis Final Project
Prep data for model training using Label2
"""
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from PadRoutes import PadRoutes

# load data
url1 = "https://raw.githubusercontent.com/parloups/csci0451-loupessis-finalproject/refs/heads/main/data/track_WR_clean.csv"
track_WR_clean = pd.read_csv(url1)

url2 = "https://raw.githubusercontent.com/parloups/csci0451-loupessis-finalproject/refs/heads/main/data/route_lables_only.csv"
labels_only = pd.read_csv(url2)

full_labels_df = labels_only.merge(track_WR_clean, on=["nflId", "frame.id", "playId"], how = "left")

# convert route lables from str to int
ROUTES = ['Post', 'Go', 'Wheel', 'Dig', 'Curl', 'Drag', 'Speed Out', 'Slot Fade', 'Slide', 'Slant', 'Crosser', 'Hitch', 
          'Whip', 'Out', 'Corner']
routes_dict = {category: idx for idx, category in enumerate(ROUTES)}
label2 = []
for _, player in full_labels_df.groupby(["playId", "nflId"]):
    route = player["Label2"].iloc[0]
    if route in routes_dict:
        label2.append(routes_dict[route])

## create feature and target vectors
X = PadRoutes(full_labels_df)
y = torch.tensor(label2)

# get proportional indexes
train_idx, test_idx = train_test_split(
    range(len(y)), test_size=0.2, stratify=y, random_state=51426
)

# save splits
X_train = torch.tensor(X[train_idx])
X_test = torch.tensor(X[test_idx])
y_train = y[train_idx].clone()
y_test = y[test_idx].clone()

# save tensors
torch.save({
    "X_train": X_train,
    "X_test": X_test,
    "y_train": y_train,
    "y_test": y_test
}, "data/tensors2.pt")