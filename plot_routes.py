"""
Parker Loupessis Final Project
Plotting/Animating the cleaned routes or entire play
Code coppied from NFL Big Data Bowl and translated to python
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from collections import defaultdict, deque

## plot WR only
track_WR_clean_df = pd.read_csv("track_WR_clean.csv")
play_id = track_WR_clean_df["playId"].unique()[24]
example_play = track_WR_clean_df[(track_WR_clean_df["playId"] == play_id)]

## plot entire play
# tracking = pd.read_csv("tracking_gameId_2017090700.csv")
# game_id = tracking.iloc[20]["gameId"]
# play_id = 1840
# example_play = tracking[(tracking["gameId"] == game_id) &
#                 (tracking["playId"] == play_id)]

xmin = 0
xmax = 160 / 3
hash_right = 38.35
hash_left  = 12
hash_width = 3.3
ymin = 0
ymax = 120

df_hash = pd.DataFrame(
    [(x, y)
     for x in [0, 23.36667, 29.96667, xmax]
     for y in range(10, 111)],
    columns=["x", "y"],
)
df_hash = df_hash[df_hash["y"] % 5 != 0]
df_hash = df_hash[(df_hash["y"] < ymax) & (df_hash["y"] > ymin)]

team_styles = {
    "home":     ("o", "#e31837", "black",   100),
    "football": ("o", "#654321", "#654321",  64),
    "away":     ("o", "#002244", "#c60c30", 100),
}

fig, ax = plt.subplots(figsize=(16, 6))
ax.set_xlim(ymin, ymax)
ax.set_ylim(xmin, xmax)
ax.set_aspect("equal")
ax.axis("off")

left_hashes  = df_hash[df_hash["x"] < 55 / 2]
right_hashes = df_hash[df_hash["x"] > 55 / 2]
ax.plot(left_hashes["y"],  left_hashes["x"],  "|", color="white", markersize=4, markeredgewidth=1.5)
ax.plot(right_hashes["y"], right_hashes["x"], "|", color="white", markersize=4, markeredgewidth=1.5)

for y in np.arange(max(10, ymin), min(ymax, 110) + 1, 5):
    ax.plot([y, y], [xmin, xmax], color="white", linewidth=0.8)

yard_labels = ["G"] + list(range(10, 50 + 1, 10)) + list(range(40, 10 - 1, -10)) + ["G"]
for i, yard_y in enumerate(range(10, 110 + 1, 10)):
    label = str(yard_labels[i])
    ax.text(yard_y, hash_left,        label, rotation=0,   fontsize=8, ha="center", va="center", color="white", fontweight="bold")
    ax.text(yard_y, xmax - hash_left, label, rotation=180, fontsize=8, ha="center", va="center", color="white", fontweight="bold")

boundary_x = [ymin, ymax, ymax, ymin, ymin]
boundary_y = [xmin, xmin, xmax, xmax, xmin]
ax.plot(boundary_x, boundary_y, color="white", linewidth=2)

ax.set_facecolor("#336633")
fig.patch.set_facecolor("#336633")

frames = sorted(example_play["frame.id"].unique())
max_players = len(example_play["nflId"].unique())

TRAIL_LENGTH = 0

position_history = defaultdict(lambda: deque(maxlen=TRAIL_LENGTH))

scatter_artists = []
text_artists    = []
trail_artists   = []
for _ in range(max_players):
    sc = ax.scatter([], [], s=100, alpha=0.7, zorder=5)
    tx = ax.text(0, 0, "", color="white", fontsize=7,
                 ha="center", va="center", zorder=6)
    line, = ax.plot([], [], linewidth=1.2, alpha=0.4, zorder=4)
    scatter_artists.append(sc)
    text_artists.append(tx)
    trail_artists.append(line)

def update(frame_id):
    frame_data = example_play[example_play["frame.id"] == frame_id]
    for idx, (_, row) in enumerate(frame_data.iterrows()):
        style = team_styles.get(row["team"], team_styles["away"])
        marker, fcolor, ecolor, size = style
        px = row["x"]
        py = xmax - row["y"]
        
        position_history[idx].append((px, py))
        if len(position_history[idx]) > 1:
            trail_x = [p[0] for p in position_history[idx]]
            trail_y = [p[1] for p in position_history[idx]]
            trail_artists[idx].set_data(trail_x, trail_y)
            trail_artists[idx].set_color(fcolor)
        else:
            trail_artists[idx].set_data([], [])

        scatter_artists[idx].set_offsets([[px, py]])
        scatter_artists[idx].set_facecolor(fcolor)
        scatter_artists[idx].set_edgecolor(ecolor)
        scatter_artists[idx].set_sizes([size])
        text_artists[idx].set_position((px, py))
        text_artists[idx].set_text(str(int(row["jerseyNumber"]))
                                   if pd.notna(row["jerseyNumber"]) else "")

    for idx in range(len(frame_data), max_players):
        scatter_artists[idx].set_offsets(np.empty((0, 2)))
        text_artists[idx].set_text("")
        trail_artists[idx].set_data([], [])

    return scatter_artists + text_artists + trail_artists

anim = FuncAnimation(fig, update, frames=frames, interval=100, blit=False)
plt.tight_layout()
plt.show()

