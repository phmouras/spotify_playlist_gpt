Spotify Playlist GPT:

A GPT-based generative recommender system trained on Spotify playlists.

This project investigates whether an autoregressive Transformer (GPT) can learn playlist structure and recommend the next tracks given a playlist context.

Pipeline: Spotify Playlists -> Corpus Builder -> Tokenizer -> Corpus.bin -> GPT Training -> Spotify API -> Recommendation Engine

The recommendation engine can evaluate offline playlists or generate recommendations for real Spotify playlists through the Spotify Web API.

First version: only URI tracks