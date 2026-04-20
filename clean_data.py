"""
Parker Loupessis Final Project Update 1
Cleaning data so labels can be added
"""

# load data
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
games = pd.read_csv('games.csv')
plays = pd.read_csv('plays.csv')
players = pd.read_csv('players.csv')
tracking = pd.read_csv('tracking_gameId_2017090700.csv')

# take only plays from game with tracking data
plays_2017090700 = plays[plays["gameId"] == 2017090700]

# merege tracking data with play details
tracking_play = tracking.merge(plays_2017090700, on="playId", how="left")

# merge tracking with player details
tracking_play_players = tracking_play.merge(players, on="nflId", how ="left")

# only keep WRs
track_WR = tracking_play_players[tracking_play_players["PositionAbbr"] == "WR"]

# remove unneeded columns
track_WR = track_WR.drop(columns=["time", "s", "dis", "dir", "gameId_y", "offenseFormation", "personnel.offense", "defendersInTheBox",
                                  "numberOfPassRushers", "personnel.defense", "HomeScoreBeforePlay", "VisitorScoreBeforePlay",
                                  "HomeScoreAfterPlay", "VisitorScoreAfterPlay", "isPenalty", "SpecialTeamsPlayType", "KickReturnYardage",
                                  "EntryYear", "DraftRound", "DraftNumber","Height", "Weight", "College", "FirstName", "LastName"])

# remove ST plays
track_WR = track_WR[track_WR["isSTPlay"] == False]

# remove designed running plays and scrambles
track_WR = track_WR[~track_WR["PassResult"].isna()]
track_WR = track_WR[track_WR["PassResult"] != "R"]

# take only the route (from the snap to pass arrives or QB is sacked)
track_WR_clean = []

for play in track_WR["playId"].unique():
  current_play = track_WR[track_WR["playId"] == play]
  for player in current_play["nflId"].unique():
    current_play_player = current_play[current_play["nflId"] == player]
    start_route = current_play_player[current_play_player["event"] == "ball_snap"]["frame.id"].iloc[0]
    end_route = current_play_player[(current_play_player["event"] == "qb_sack") |
                                    (current_play_player["event"] == "pass_arrived") |
                                    (current_play_player["event"] == "first_contact") |
                                    (current_play_player["event"] == "pass_outcome_incomplete") |
                                    (current_play_player["event"] == "pass_outcome_caught") |
                                    (current_play_player["event"] == "pass_outcome_touchdown")
                                    ]["frame.id"].iloc[0]

    current_route = current_play_player[(current_play_player["frame.id"] >= start_route) &
                                        (current_play_player["frame.id"] <= end_route)]
    track_WR_clean.append(current_route)

track_WR_clean_df = pd.concat(track_WR_clean).reset_index(drop=True)

track_WR_clean_df.head()


# Pick a play
play_id = track_WR_clean_df["playId"].unique()[25]
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
    
    # Only look at recent frames (prevents lag)
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
    
    # Current positions (same as before)
    offense = frame_data[frame_data["team"] == "home"]
    defense = frame_data[frame_data["team"] == "away"]
    football = frame_data[frame_data["team"] == "football"]
    
    ax.scatter(offense["x"], offense["y"], c="blue", s=50)
    ax.scatter(defense["x"], defense["y"], c="red", s=50)
    ax.scatter(football["x"], football["y"], c="brown", s=70)
    
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 53.3)
    ax.set_title(f"Game 2017090700, Play {play_id}, Frame {frame}")

ani = animation.FuncAnimation(fig, update, frames=frames, interval=100)

plt.show()