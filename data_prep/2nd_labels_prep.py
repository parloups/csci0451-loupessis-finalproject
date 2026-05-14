"""
Parker Loupessis Final Project
Prep Data for Model 4
"""
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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
labels2 = []
for _, player in full_labels_df.groupby(["playId", "nflId"]):
    route = player["Label2"].iloc[0]
    if route in routes_dict:
        labels2.append(routes_dict[route])

# pad routes to all have same number of frames 
max_frames = full_labels_df["frame.id"].max()
padded_sequences = []

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

  direction_rad = current_dir_values * (np.pi / 180)
  dir_sin = np.sin(direction_rad)
  dir_cos = np.cos(direction_rad)  

  current_len = len(player_route_df)
  len_seq = np.full(max_frames, current_len)

  if current_len < max_frames:
    pad_len = max_frames - current_len
    padded_x = np.pad(current_x_values, (0, pad_len), "constant", constant_values=0)
    padded_y = np.pad(current_y_values, (0, pad_len), "constant", constant_values=0)
    padded_s = np.pad(current_s_values, (0, pad_len), "constant", constant_values=0)
    padded_dis = np.pad(current_dis_values, (0, pad_len), "constant", constant_values=0)
    padded_dir_sin = np.pad(dir_sin, (0, pad_len), "constant", constant_values=0)
    padded_dir_cos = np.pad(dir_cos, (0, pad_len), "constant", constant_values=0)
  else:
    padded_x = current_x_values
    padded_y = current_y_values
    padded_s = current_s_values
    padded_dis = current_dis_values
    padded_dir = current_dir_values
    padded_dir_sin = dir_sin
    padded_dir_cos = dir_cos

  relative_x = padded_x - snap_x_seq
  relative_y = padded_y - snap_y_seq

  combined_sequence = np.stack([padded_x, padded_y, snap_x_seq, snap_y_seq, relative_x, relative_y, padded_s, padded_dir_sin, padded_dir_cos, padded_dis, len_seq], axis = 1)
  padded_sequences.append(combined_sequence)

all_padded_routes = np.array(padded_sequences)

## create feature and target vectors
X = all_padded_routes
y = torch.tensor(labels2)

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