"""
Parker Loupessis Final Project
Cleaning data so labels can be added
"""

# load data
import pandas as pd
url = "https://raw.githubusercontent.com/parloups/csci0451-loupessis-finalproject/refs/heads/main/data/games.csv"
games = df = pd.read_csv(url)

url2 = "https://raw.githubusercontent.com/parloups/csci0451-loupessis-finalproject/refs/heads/main/data/players.csv"
players = pd.read_csv(url2)

url3 = "https://raw.githubusercontent.com/parloups/csci0451-loupessis-finalproject/refs/heads/main/data/plays.csv"
plays = pd.read_csv(url3)

url4 = "https://raw.githubusercontent.com/parloups/csci0451-loupessis-finalproject/refs/heads/main/data/tracking_gameId_2017090700.csv"
tracking = pd.read_csv(url4)

# take only plays from game with tracking data
plays_2017090700 = plays[plays["gameId"] == 2017090700]

# merege tracking data with play details
tracking_play = tracking.merge(plays_2017090700, on="playId", how="left")

# merge tracking with player details
tracking_play_players = tracking_play.merge(players, on="nflId", how ="left")

# only keep WRs
track_WR = tracking_play_players[tracking_play_players["PositionAbbr"] == "WR"]

# remove unneeded columns
track_WR = track_WR.drop(columns=["time", "gameId_y", "offenseFormation", "personnel.offense", "defendersInTheBox",
                                  "numberOfPassRushers", "personnel.defense", "HomeScoreBeforePlay", "VisitorScoreBeforePlay",
                                  "HomeScoreAfterPlay", "VisitorScoreAfterPlay", "isPenalty", "SpecialTeamsPlayType", "KickReturnYardage",
                                  "EntryYear", "DraftRound", "DraftNumber","Height", "Weight", "College", "FirstName", "LastName"])

# remove ST plays
track_WR = track_WR[track_WR["isSTPlay"] == False]

# remove designed running plays and scrambles
track_WR = track_WR[~track_WR["PassResult"].isna()]
track_WR = track_WR[track_WR["PassResult"] != "R"]

# remove shovel passes
shovel_event = track_WR[track_WR["event"] == "pass_shovel"]
shovel_plays = shovel_event["playId"].unique()
track_WR = track_WR[~track_WR["playId"].isin(shovel_plays)]

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

# Get the snap location from the football's (x,y) before the snap
football = tracking[tracking["displayName"] == "football"]
snap_xy = []
for snap in football["playId"].unique():
  x_snap = football[football["playId"] == snap]["x"].iloc[0].item()
  y_snap = football[football["playId"] == snap]["y"].iloc[0].item()
  snap_xy.append((x_snap, y_snap, snap.item()))

snap_xy_df = pd.DataFrame(snap_xy, columns=['snap_x', 'snap_y', 'playId'])

# Add snap location to WR df
track_WR_clean_df = track_WR_clean_df.merge(snap_xy_df, on="playId", how ="left")

# Save df as csv
track_WR_clean_df.to_csv('track_WR_clean.csv', index=False)