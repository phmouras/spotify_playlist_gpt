"""
- corpus.bin
- indx_part.csv
- vocab.pkl
- vocab_size.txt
"""

import argparse
import logging
import pickle
from pathlib import Path
import sys

import pandas as pd
import numpy as np

from tokenizador_playlist_spotify import load_playlists_from_json, build_global_vocabs, encode_playlist


parser = argparse.ArgumentParser()

parser.add_argument("--output_dir",type=str, default="./corpus")
parser.add_argument("--max_files", type=int, default=10)

args = parser.parse_args()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

UNKNOWN_TRACK_TOKEN = 1
FIRST_TRACK_TOKEN = 100

logger = logging.getLogger("build_corpus")

json_dir = Path("./spotify-playlist-dataset/data")

output_dir = Path(args.output_dir)

output_dir.mkdir(parents=True, exist_ok=True)

logger.info("Loading playlists...")

try:        

    dataframe_chunks = []

    for index, df_chunks in enumerate(load_playlists_from_json(
        json_path=json_dir,
        max_files=args.max_files,
        chunk_size=500_000,
        selected_columns=[
            "pid",
            "pos",
            "track_uri"]), start=1):

        logger.info("Chunk %s: %s rows loaded", index, f"{len(df_chunks):,}")

        dataframe_chunks.append(df_chunks)  

    logger.info("Combining DataFrame chunks...")

    df = pd.concat(dataframe_chunks, ignore_index=True)

    del dataframe_chunks

    # RAM usage in MB
    memory_mb = (df.memory_usage(index=True, deep=True).sum()/1024**2)

    logger.info(
        "Final DataFrame memory usage: %.2f MB",
        memory_mb,
    )

except Exception:
    logger.exception("Error to load playlists")
    sys.exit(1)

logger.info("%s rows loaded:", f"{len(df):,}")
logger.info("%s playlists found:", f"{df['pid'].nunique():,}")
logger.info("%s unique tracks found:", f"{df['track_uri'].nunique():,}")
logger.info("Building global vocabulary...")

vocabs, vocab_size = build_global_vocabs(
    json_path=json_dir,
    max_files=args.max_files,
)

logger.info("Encoding playlists...")

vf, indx_part = encode_playlist(df, vocabs=vocabs)

logger.info("Validating corpus.bin...")

# validate corpus    
if len(vf) == 0:
    raise ValueError("The generated corpus is empty.")

if len(indx_part) == 0:
    raise ValueError("No playlists were indexed.")

if vf.max() >= vocab_size:
    raise ValueError(f"Token {vf.max()} exceeds vocab_size={vocab_size}")

vf.astype(np.uint32).tofile(output_dir/"corpus.bin")

logger.info("Saving stats.csv...")

indx_part.to_csv(output_dir / "stats.csv", index=False)

logger.info("Saving vocabulary...")

with open(output_dir / "vocab.pkl", "wb",) as file:
    pickle.dump(vocabs, file) # serialize as binary

logger.info("Saving vocab_size...")

with open(output_dir / "vocab_size.txt", "w", encoding="utf-8") as file:
    file.write(str(vocab_size))

logger.info("%s tokens generated.", f"{len(vf):,}")
logger.info("%s playlists indexed.", f"{len(indx_part):,}")
logger.info("vocab_size = %s", f"{vocab_size:,}")

print("Finish")