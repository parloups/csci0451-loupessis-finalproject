"""
Parker Loupessis Final Project Update 1
Plotting/Animating the cleaned routes
"""
import matplotlib.pyplot as plt
from matplotlib import animation
from clean_data import track_WR_clean_df

# Pick a play
play_id = track_WR_clean_df["playId"].unique()[29]
play = track_WR_clean_df[(track_WR_clean_df["playId"] == play_id)]

# Sort by frame
play = play.sort_values("frame.id")

# Get unique frames
frames = play["frame.id"].unique()

# Set up plot
fig, ax = plt.subplots(figsize=(12,6))

def update(frame):
    ax.clear()
    
    # Current frame
    frame_data = play[play["frame.id"] == frame]
    
    # Only look at recent frames
    trail_length = 100
    min_frame = max(frame - trail_length, play["frame.id"].min())
    history = play[(play["frame.id"] >= min_frame) & (play["frame.id"] <= frame)]
    
    # Draw trail per player
    for nflId in frame_data["nflId"].unique():
        player_hist = history[history["nflId"] == nflId]
        
        if len(player_hist) < 2:
            continue  # need at least 2 points to draw a line
        
        team = player_hist["team"].iloc[0]
        if team == "home":
            color = "blue"
        elif team == "away":
            color = "red"
        else:
            color = "brown"
        
        ax.plot(player_hist["x"], player_hist["y"], color=color, alpha=0.6)
    
    # Current positions
    offense = frame_data[frame_data["team"] == "home"]
    defense = frame_data[frame_data["team"] == "away"]
    #football = frame_data[frame_data["team"] == "football"]
    
    ax.scatter(offense["x"], offense["y"], c="blue", s=50)
    ax.scatter(defense["x"], defense["y"], c="red", s=50)
    #ax.scatter(football["x"], football["y"], c="brown", s=70)
    
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 53.3)
    ax.set_title(f"Game 2017090700, Play {play_id}, Frame {frame}")

ani = animation.FuncAnimation(fig, update, frames=frames, interval=100)

plt.show()