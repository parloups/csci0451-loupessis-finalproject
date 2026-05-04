"""
Parker Loupessis Final Project
Prep Data for Model Training
"""
import torch
import pandas as pd
import numpy as np

# load data
url1 = "https://raw.githubusercontent.com/parloups/csci0451-loupessis-finalproject/refs/heads/main/data/track_WR_clean.csv"
track_WR_clean = pd.read_csv(url1)

url2 = "https://raw.githubusercontent.com/parloups/csci0451-loupessis-finalproject/refs/heads/main/data/route_lables_only.csv"
labels_only = pd.read_csv(url2)

full_labels_df = labels_only.merge(track_WR_clean, on=["nflId", "frame.id", "playId"], how = "left")

# convert route lables from str to int
ROUTES = ["Go", "In", "Out", "Curl"]
routes_dict = {category: idx for idx, category in enumerate(ROUTES)}
labels1 = []
for _, player in full_labels_df.groupby(["playId", "nflId"]):
    route = player["Label1"].iloc[0]
    if route in routes_dict:
        labels1.append(routes_dict[route])

# pad routes to all have same number of frames 
max_frames = full_labels_df["frame.id"].max()
padded_sequences = []
orig_lengths = []

for _, player_route_df in full_labels_df.groupby(["playId", "nflId"]):
  snap_x = player_route_df["snap_x"].iloc[0]
  snap_y = player_route_df["snap_y"].iloc[0]
  snap_x_seq = np.full(max_frames, snap_x)
  snap_y_seq = np.full(max_frames, snap_y)

  current_x_values = player_route_df["x"].values
  current_y_values = player_route_df["y"].values
  current_s_values = player_route_df["s"].values
  current_dis_values = player_route_df["dis"].values
  current_dir_values = player_route_df["dir"].values

  current_len = len(player_route_df)
  len_seq = np.full(max_frames, current_len)

  if current_len < max_frames:
    pad_len = max_frames - current_len
    padded_x = np.pad(current_x_values, (0, pad_len), "constant", constant_values=0)
    padded_y = np.pad(current_y_values, (0, pad_len), "constant", constant_values=0)
    padded_s = np.pad(current_s_values, (0, pad_len), "constant", constant_values=0)
    padded_dis = np.pad(current_dis_values, (0, pad_len), "constant", constant_values=0)
    padded_dir = np.pad(current_dir_values, (0, pad_len), "constant", constant_values=0)
  else:
    padded_x = current_x_values
    padded_y = current_y_values
    padded_s = current_s_values
    padded_dis = current_dis_values
    padded_dir = current_dir_values

  combined_sequence = np.stack([padded_x, padded_y, snap_x_seq, snap_y_seq, padded_s, padded_dis, padded_dir, len_seq], axis = 1)
  padded_sequences.append(combined_sequence)

all_padded_routes = np.array(padded_sequences)

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

# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

torch.manual_seed(50426)
## create validation set
route_counts = torch.bincount(y_train)
val_data_counts = (route_counts * 0.2).round()
# Go routes
idx0 = torch.where(y_train == 0)[0]
shuff0 = idx0[torch.randperm(len(idx0))]
trainidx0 = shuff0[int(val_data_counts[0].item()):]
validx0 = shuff0[:int(val_data_counts[0].item())]

# In routes
idx1 = torch.where(y_train == 1)[0]
shuff1 = idx1[torch.randperm(len(idx1))]
trainidx1 = shuff1[int(val_data_counts[1].item()):]
validx1 = shuff1[:int(val_data_counts[1].item())]

# Out routes
idx2 = torch.where(y_train == 2)[0]
shuff2= idx2[torch.randperm(len(idx2))]
trainidx2 = shuff2[int(val_data_counts[2].item()):]
validx2 = shuff2[:int(val_data_counts[2].item())]

# Curl routes
idx3 = torch.where(y_train == 3)[0]
shuff3 = idx3[torch.randperm(len(idx3))]
trainidx3 = shuff3[int(val_data_counts[3].item()):]
validx3 = shuff3[:int(val_data_counts[3].item())]

# All routes
trainidx = torch.concat([trainidx0, trainidx1, trainidx2, trainidx3])
validx = torch.concat([validx0, validx1, validx2, validx3])

# Concat 
X_train_list = []
X_val_list = []
y_train_list = []
y_val_list = []
for i in trainidx:
  X_train_list.append(X_train[i])
  y_train_list.append(y_train[i].item())
for i in validx:
  X_val_list.append(X_train[i])
  y_val_list.append(y_train[i].item())

X_train = torch.tensor(np.stack(X_train_list, axis = 0))
X_val = torch.tensor(np.stack(X_val_list, axis = 0))
y_train = torch.tensor(y_train_list)
y_val = torch.tensor(y_val_list)

print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)