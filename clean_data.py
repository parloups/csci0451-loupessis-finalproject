"""
Parker Loupessis Final Project Update 1
Cleaning data so labels can be added
"""

# load data
import pandas as pd
import numpy as np
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