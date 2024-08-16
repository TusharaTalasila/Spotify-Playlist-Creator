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

def extract_features(df):
    numeric_features = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
                        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

    le = LabelEncoder()
    df['main_artist_encoded'] = le.fit_transform(df['main_artist'])

    features = df[numeric_features + ['main_artist_encoded']]

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    return scaled_features, scaler, le


def apply_incremental_pca(features, n_components=None):
    n_features = features.shape[1]
    if n_components is None or n_components > n_features:
        n_components = n_features - 1

    ipca = IncrementalPCA(n_components=n_components, batch_size=1000)
    return ipca.fit_transform(features), ipca

def cluster_tracks(pca_features, n_clusters=10):
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1000)
    return kmeans.fit_predict(pca_features)