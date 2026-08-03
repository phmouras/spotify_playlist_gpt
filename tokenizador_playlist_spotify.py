# tokenizador playlist spotify

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


json_path = Path("./spotify-playlist-dataset/data")

# Load playlists from json files into a df. (read chunks)
def load_playlists_from_json(
    json_path,
    max_files,
    chunk_size,
    selected_columns: list[str] | None = None):

    files = sorted(json_path.glob("*.json")) # sort the files to ensure consistent order

    if max_files is not None:
        files = files[:max_files]

    if selected_columns is None:
        selected_columns = [
            "pid",
            "playlist_name",
            "pos",
            "track_name",
            "artist_name",
            "album_name",
            "track_uri",
            "artist_uri",
            "album_uri",
            "duration_ms"
        ]

    rows = [] 

    for file in files:
        with open(file, "r", encoding="utf-8") as f: # read the JSON file with UTF-8 encoding 
            data = json.load(f) 

        for playlist in data["playlists"]:

            playlist_data = {
                "pid": playlist["pid"], 
                "playlist_name": playlist["name"]}

            for track in playlist["tracks"]: # collect track information
                track_data = {
                    "pos": track["pos"],
                    "track_name": track["track_name"],
                    "artist_name": track["artist_name"],
                    "album_name": track["album_name"],
                    "track_uri": track["track_uri"],
                    "artist_uri": track["artist_uri"],
                    "album_uri": track["album_uri"],
                    "duration_ms": track["duration_ms"],
                }

                full_row = {**playlist_data, **track_data} # merge playlist and track data into a single row

                filtered_row = {columns: full_row.get(columns) for columns in selected_columns}

                rows.append(filtered_row) # 1 row per track

            if len(rows) >= chunk_size:
                yield pd.DataFrame(rows)
                rows = []

        del data
    
    if rows: 
        yield pd.DataFrame(rows) # create a DataFrame from the remaining rows if any

UNKNOWN_TRACK_TOKEN = 1
FIRST_TRACK_TOKEN = 100

# corpus sequence
tk_columns = ['tk_new_playlist','tk_track_uri'
            #   'c_artist_uri', 'tk_artist_uri',
            #   'c_album_uri', 'tk_album_uri'
            ]

# w/ chunks
def build_global_vocabs(json_path, max_files=None, chunk_size=100_000, categorical_columns=None): 
    """
    Global vocabs for categorical columns

        vocabs:
            mapping URI -> token and token -> URI for each categorical column, along with the token ranges

        tokens_ranges:
            dict with the start, end, and size of the token range for each categorical column
    """

    if categorical_columns is None:
        categorical_columns = [
            "track_uri",
            # "artist_uri",
            # "album_uri"
        ]

    unique_values = {column: set() for column in categorical_columns}  # dict to store unique values for each categorical column

    selected_columns = ["pid", "pos", *categorical_columns]

    for df_chunk in load_playlists_from_json(
        json_path=json_path,
        max_files=max_files,
        chunk_size=chunk_size,
        selected_columns=selected_columns):

        for column in categorical_columns:
            values = (df_chunk[column].dropna().astype(str).unique())
            unique_values[column].update(values)

    vocabs = {}         # dict to store global vocab information for each categorical column

    # 0: delimitador - new playlist
    # 1: unknown artist token
    # 2 .. 99 : reserved 
    first_token = FIRST_TRACK_TOKEN

    for col in categorical_columns:

        ordered_values = sorted(unique_values[col])  # sort the unique values for consistent token assignment

        # {uri: token} and {token: uri} mappings for the current column
        uri_to_token = {                                
            uri: first_token + index   # 100, 101, 102, ... uri_n            
            for index, uri in enumerate(ordered_values)  # 0 1 ... , uri_0, uri_1, ...
        }

        token_to_uri = {
            token: uri
            for uri, token in uri_to_token.items()
        }

        # Store the vocab and token range for the current column
        vocabs[col] = {                     
            "uri_to_token": uri_to_token,
            "token_to_uri": token_to_uri,
        }

        first_token += len(ordered_values) # no subscribe to the next token 

    vocab_size = first_token

    return vocabs, vocab_size


# no values columns

def encode_playlist(df, vocabs):

    categoric_columns = ["track_uri"]
    
    track_to_token = vocabs["track_uri"]["uri_to_token"]  # uri -> token

    for col in categoric_columns:
        df[f"tk_{col}"] = df[col].map(track_to_token).fillna(UNKNOWN_TRACK_TOKEN).astype(np.int32)  # 1 for unknown tracks

    # We have only track tokens, so we dont need marks 
    # marks
    # df["c_artist_name"] = 1
    # df["c_album_name"] = 2
    # df["c_track_name"] = 3

    primeiros = df.groupby('pid').pos.min().reset_index().copy()
    primeiros['tk_new_playlist'] = 0
    df_quebra = pd.concat([df, primeiros], ignore_index=True).sort_values(['pid', 'pos', 'tk_new_playlist'])
    vf = df_quebra[tk_columns].values.reshape(len(df_quebra)*len(tk_columns))
    vf = vf[~np.isnan(vf)].astype(np.int32).copy() # vetor tokens
    
    quebra_dossies = np.concatenate([np.nonzero(vf == 0)[0], [len(vf)]]) # vetor de indices de quebra de dossies
    indx_part = df[['pid']].drop_duplicates().sort_values('pid').reset_index(drop=True).copy()  
    indx_part['inicio'] = quebra_dossies[0:-1] + 1
    indx_part['fim'] = quebra_dossies[1:]
    indx_part["num_tracks"] = (indx_part["fim"] - indx_part["inicio"])

    return vf, indx_part
