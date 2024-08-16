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
from feature_engineering import extract_features

def build_recommender(pca_features):
    nn = NearestNeighbors(n_neighbors=10, metric='cosine', algorithm='brute')
    return nn.fit(pca_features)


def get_recommendations(user_tracks, dataset, recommender, ipca, scaler, le, n=5):
    try:
        print(f"Shape of user_tracks: {user_tracks.shape}")
        user_features, _, _ = extract_features(user_tracks)
        print(f"Shape of user_features: {user_features.shape}")
        user_features_scaled = scaler.transform(user_features)
        print(f"Shape of user_features_scaled: {user_features_scaled.shape}")
        user_pca = ipca.transform(user_features_scaled)
        print(f"Shape of user_pca: {user_pca.shape}")

        distances, indices = recommender.kneighbors(user_pca)
        print(f"Shape of distances: {distances.shape}, Shape of indices: {indices.shape}")

        # Flatten and remove duplicates while preserving order
        unique_indices = []
        seen = set()
        for idx_array in indices:
            for idx in idx_array:
                if idx not in seen and idx < len(dataset):
                    seen.add(idx)
                    unique_indices.append(idx)
                if len(unique_indices) == n:
                    break
            if len(unique_indices) == n:
                break

        print(f"Number of unique indices: {len(unique_indices)}")
        print(f"Unique indices: {unique_indices}")

        if len(unique_indices) == 0:
            print("No valid recommendations found. Using random sampling as fallback.")
            unique_indices = np.random.choice(len(dataset), size=min(n, len(dataset)), replace=False)

        recommendations = dataset.iloc[unique_indices]

        recommendations['artists'] = recommendations['artists'].apply(
            lambda x: ', '.join(x) if isinstance(x, list) else x)

        return recommendations[['name', 'artists', 'album', 'id']]
    except Exception as e:
        print(f"Error in get_recommendations: {str(e)}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(columns=['name', 'artists', 'album', 'id'])