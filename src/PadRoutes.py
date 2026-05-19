"""
Parker Loupessis Final Project
Function to pad routes to same length for training
"""
import numpy as np

def PadRoutes(df):
  """
  Pads all player routes to the same length so they can be batched for training.
    
  Each route is padded with zeros at the end up to the length of the longest
  route in the dataset. The real route length is stored as a feature so the
  model can ignore padded timesteps via pack_padded_sequence.

  Args:
    df: DataFrame containing tracking data with columns for playId, nflId, frame.id, x, y, s, dis, dir, snap_x, snap_y

  Returns:
    np.array of shape (num_routes, max_frames, 11) where the 11 features are:
      [x, y, snap_x, snap_y, relative_x, relative_y, s, dir_sin, dir_cos, dis, length]
    """
  # find the longest route in the dataset — all routes will be padded to this length
  max_frames = df["frame.id"].max()
  padded_sequences = []

  for _, player_route_df in df.groupby(["playId", "nflId"]):
    # snap location is constant for the whole play — tile it across all timesteps
    snap_x = player_route_df["snap_x"].iloc[0]
    snap_y = player_route_df["snap_y"].iloc[0]
    snap_x_seq = np.full(max_frames, snap_x)
    snap_y_seq = np.full(max_frames, snap_y)

    # extract raw tracking features for this route
    current_x_values = player_route_df["x"].values
    current_y_values = player_route_df["y"].values
    current_s_values = player_route_df["s"].values
    current_dis_values = player_route_df["dis"].values
    current_dir_values = player_route_df["dir"].values

    # convert direction from degrees to sin/cos to avoid discontinuity at 0/360
    # code suggested by AI
    direction_rad = current_dir_values * (np.pi / 180)
    dir_sin = np.sin(direction_rad)
    dir_cos = np.cos(direction_rad)  

    # store real route length
    current_len = len(player_route_df)
    len_seq = np.full(max_frames, current_len)

    # pad routes with zeros at the end up to max_frames
    if current_len < max_frames:
      pad_len = max_frames - current_len
      padded_x = np.pad(current_x_values, (0, pad_len), "constant", constant_values=0)
      padded_y = np.pad(current_y_values, (0, pad_len), "constant", constant_values=0)
      padded_s = np.pad(current_s_values, (0, pad_len), "constant", constant_values=0)
      padded_dis = np.pad(current_dis_values, (0, pad_len), "constant", constant_values=0)
      padded_dir_sin = np.pad(dir_sin, (0, pad_len), "constant", constant_values=0)
      padded_dir_cos = np.pad(dir_cos, (0, pad_len), "constant", constant_values=0)
    else:
      # route is already at max length — no padding needed
      padded_x = current_x_values
      padded_y = current_y_values
      padded_s = current_s_values
      padded_dis = current_dis_values
      padded_dir_sin = dir_sin
      padded_dir_cos = dir_cos

    # compute position relative to snap location
    relative_x = padded_x - snap_x_seq
    relative_y = padded_y - snap_y_seq

    # stack all features into a single array of shape
    combined_sequence = np.stack([padded_x, padded_y, snap_x_seq, snap_y_seq, relative_x, relative_y, padded_s, padded_dir_sin, padded_dir_cos, padded_dis, len_seq], axis = 1)
    padded_sequences.append(combined_sequence)

  return np.array(padded_sequences)