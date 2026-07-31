# tokenizador playlist spotify

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


json_path = Path("./spotify-playlist-dataset/data")

# Load playlists from json files into a df
def load_playlists_from_json(json_path, max_files=None):
    
    files = sorted(json_path.glob("*.json")) # sort the files to ensure consistent order

    if max_files is not None:
        files = files[:max_files]

    rows = [] 

    for file in files:
        with open(file, "r", encoding="utf-8") as f: # read the JSON file with UTF-8 encoding 
            data = json.load(f) 

        for playlist in data["playlists"]:
            pid = playlist["pid"] # collect playlist ID
            playlist_name = playlist.get("name", "") # collect playlist name if available, else empty string

            for track in playlist["tracks"]: # collect track information
                rows.append({
                    "pid": pid,
                    "playlist_name": playlist_name,
                    "pos": track["pos"],
                    "track_name": track["track_name"],
                    "artist_name": track["artist_name"],
                    "album_name": track["album_name"],
                    "track_uri": track["track_uri"],
                    "artist_uri": track["artist_uri"],
                    "album_uri": track["album_uri"],
                    "duration_ms": track["duration_ms"],
                })

    return pd.DataFrame(rows)

df = load_playlists_from_json(
    json_path, max_files=3
)

UNKNOWN_TRACK_TOKEN = 1
FIRST_TRACK_TOKEN = 100

# corpus sequence
tk_columns = ['tk_new_playlist','tk_track_uri'
            #   'c_artist_uri', 'tk_artist_uri',
            #   'c_album_uri', 'tk_album_uri'
            ]


def build_global_vocabs(df):
    """
    Global vocabs for categorical columns

        vocabs:
            mapping URI -> token and token -> URI for each categorical column, along with the token ranges

        tokens_ranges:
            dict with the start, end, and size of the token range for each categorical column
    """

    categorical_columns = [
        "track_uri",
        # "artist_uri",
        # "album_uri",
    ]

    vocabs = {}         # dict to store global vocab information for each categorical column

    # 0: delimitador - new playlist
    # 1: unknown artist token
    # 2 .. 99 : reserved 
    first_token = FIRST_TRACK_TOKEN

    for col in categorical_columns:

        # Order vocabulary by URI to ensure consistent token assignment across different runs
        unique_values = sorted(df[col].dropna().astype(str).unique().tolist())

        # {uri: token} and {token: uri} mappings for the current column
        uri_to_token = {                                
            uri: first_token + index   # 100, 101, 102, ... uri_n            
            for index, uri in enumerate(unique_values)  # 0 1 ... , uri_0, uri_1, ...
        }

        token_to_uri = {
            token: uri
            for uri, token in uri_to_token.items()
        }

        start = first_token
        end = first_token + len(unique_values) - 1

        vocab_size = end + 1

        # Store the vocab and token range for the current column
        vocabs[col] = {                     
            "uri_to_token": uri_to_token,
            "token_to_uri": token_to_uri,
        }

        ## good for more than 1 column to be tokenized
        # token_ranges[col] = {
        #     "start": start,
        #     "end": end,
        #     "size": len(unique_values),
        # }

        first_token = end + 1 # no subscribe to the next token 

    vocab_size = first_token

    return vocabs, vocab_size


# no values columns

def encode_playlist(df, vocabs):
    df = df.copy()

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
    print(quebra_dossies) 
    indx_part = df[['pid']].drop_duplicates().sort_values('pid').reset_index(drop=True).copy()  
    indx_part['inicio'] = quebra_dossies[0:-1] + 1
    indx_part['fim'] = quebra_dossies[1:]
    indx_part["num_tracks"] = (indx_part["fim"] - indx_part["inicio"])

    return vf, indx_part
