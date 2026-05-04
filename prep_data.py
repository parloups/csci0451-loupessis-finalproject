"""
Parker Loupessis Final Project
Prep Data for Model Training
"""
import torch
import pandas as pd
import numpy as np

# load data
url = "https://raw.githubusercontent.com/parloups/csci0451-loupessis-finalproject/refs/heads/main/data/track_WR_labels.csv"
wr_labels = df = pd.read_csv(url)

# convert route lables from str to int
ROUTES = ["Go", "In", "Out", "Curl"]
routes_dict = {category: idx for idx, category in enumerate(ROUTES)}
labels1 = []
for _, player in wr_labels.groupby(["playId", "nflId"]):
    route = player["Label1"].iloc[0]
    if route in routes_dict:
        labels1.append(routes_dict[route])

# pad routes to all have same number of frames 
max_frames = wr_labels["frame.id"].max()
padded_sequences = []
orig_lengths = []

for _, player_route_df in wr_labels.groupby(["playId", "nflId"]):
  snap_x = player_route_df["snap_x"].iloc[0]
  snap_y = player_route_df["snap_y"].iloc[0]
  snap_x_seq = np.full(max_frames, snap_x)
  snap_y_seq = np.full(max_frames, snap_y)

  current_x_values = player_route_df["x"].values
  current_y_values = player_route_df["y"].values

  current_len = len(player_route_df)
  orig_lengths.append(current_len)

  if current_len < max_frames:
    pad_len = max_frames - current_len
    padded_x = np.pad(current_x_values, (0,pad_len), "constant", constant_values=0)
    padded_y = np.pad(current_y_values, (0, pad_len), "constant", constant_values=0)
  else:
    padded_x = current_x_values
    padded_y = current_y_values

  combined_sequence = np.stack([padded_x, padded_y, snap_x_seq, snap_y_seq], axis = 1)
  padded_sequences.append(combined_sequence)

all_padded_routes = np.array(padded_sequences)
lengths = torch.tensor(orig_lengths)

## create feature and target vectors
X = all_padded_routes
y = torch.tensor(labels1)

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

#print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
