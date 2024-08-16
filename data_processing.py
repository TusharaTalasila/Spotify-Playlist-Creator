import os
import time
import random
from spotipy.oauth2 import SpotifyOAuth
import spotipy
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
import ast

def load_dataset(file_path, chunk_size=10000):
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        chunk['artists'] = chunk['artists'].apply(ast.literal_eval)
        chunk['main_artist'] = chunk['artists'].apply(lambda x: x[0] if x else '')
        yield chunk


def match_tracks_with_dataset(tracks, dataset_chunk):
    matched_tracks = []
    for track in tracks:
        match = dataset_chunk[
            (dataset_chunk['name'].str.lower() == track['name'].lower()) &
            (dataset_chunk['main_artist'].str.lower() == track['artists'][0]['name'].lower())
            ]
        if not match.empty:
            matched_tracks.append(match.iloc[0])
    return pd.DataFrame(matched_tracks)