"""
Parker Loupessis Final Project
Prep data for model training using Label1
"""
import torch
import pandas as pd
import numpy as np
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
ROUTES = ["Go", "In", "Out", "Curl"]
routes_dict = {category: idx for idx, category in enumerate(ROUTES)}
label1 = []
for _, player in full_labels_df.groupby(["playId", "nflId"]):
    route = player["Label1"].iloc[0]
    if route in routes_dict:
        label1.append(routes_dict[route])

## create feature and target vectors
X = PadRoutes(full_labels_df)
y = torch.tensor(label1)

# create testing data with same proportion of each route represented
route_counts = torch.bincount(y)
test_data_counts = (route_counts * 0.2).round()

torch.manual_seed(43026)

# Go routes
idx0 = torch.where(y == 0)[0]
shuff0 = idx0[torch.randperm(len(idx0))]
trainidx0 = shuff0[int(test_data_counts[0].item()):]
testidx0 = shuff0[:int(test_data_counts[0].item())]

# In routes
idx1 = torch.where(y == 1)[0]
shuff1 = idx1[torch.randperm(len(idx1))]
trainidx1 = shuff1[int(test_data_counts[1].item()):]
testidx1 = shuff1[:int(test_data_counts[1].item())]

# Out routes
idx2 = torch.where(y == 2)[0]
shuff2= idx2[torch.randperm(len(idx2))]
trainidx2 = shuff2[int(test_data_counts[2].item()):]
testidx2 = shuff2[:int(test_data_counts[2].item())]

# Curl routes
idx3 = torch.where(y == 3)[0]
shuff3 = idx3[torch.randperm(len(idx3))]
trainidx3 = shuff3[int(test_data_counts[3].item()):]
testidx3 = shuff3[:int(test_data_counts[3].item())]

# All routes
trainidx = torch.concat([trainidx0, trainidx1, trainidx2, trainidx3])
testidx = torch.concat([testidx0, testidx1, testidx2, testidx3])

# Concat 
X_train_list = []
X_test_list = []
y_train_list = []
y_test_list = []
for i in trainidx:
  X_train_list.append(X[i])
  y_train_list.append(y[i].item())
for i in testidx:
  X_test_list.append(X[i])
  y_test_list.append(y[i].item())

X_train = torch.tensor(np.stack(X_train_list, axis = 0))
X_test = torch.tensor(np.stack(X_test_list, axis = 0))
y_train = torch.tensor(y_train_list)
y_test = torch.tensor(y_test_list)

# save tensors
torch.save({
    "X_train": X_train,
    "X_test": X_test,
    "y_train": y_train,
    "y_test": y_test
}, "data/tensors.pt")