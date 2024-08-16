import os
import time
import random
from pydoc import importfile

from spotipy.oauth2 import SpotifyOAuth
import spotipy
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
import ast
from data_processing import load_dataset, match_tracks_with_dataset
from feature_engineering import extract_features, apply_incremental_pca, cluster_tracks
from model import build_recommender, get_recommendations
from spotify_utils import (
    get_spotify_client,
    get_user_playlists,
    get_playlist_tracks,
    create_playlist
)

def main():
    sp = get_spotify_client()

    try:
        user = sp.current_user()
        print(f"Successfully authenticated as {user['display_name']}")

        user_id = user['id']
        playlists = get_user_playlists(sp, user_id)

        print("Processing dataset...")
        all_features = []
        total_tracks = 0
        scaler = None
        le = None

        for chunk in load_dataset('spotify_tracks.csv'):
            features, scaler, le = extract_features(chunk)
            all_features.append(features)
            total_tracks += len(chunk)
            print(f"Processed {total_tracks} tracks...")

        all_features = np.vstack(all_features)
        print("Applying Incremental PCA...")
        pca_features, ipca = apply_incremental_pca(all_features)
        print("PCA features shape:", pca_features.shape)

        print("Clustering tracks...")
        clusters = cluster_tracks(pca_features)
        print("Clusters shape:", clusters.shape)

        print("Building recommender...")
        recommender = build_recommender(pca_features)

        print("\nYour playlists:")
        for i, playlist in enumerate(playlists, 1):
            print(f"{i}. {playlist['name']} (Tracks: {playlist['tracks']['total']})")

        playlist_choice = int(input("\nEnter the number of the playlist you want to use for recommendations: ")) - 1
        selected_playlist = playlists[playlist_choice]

        print(f"\nProcessing playlist: {selected_playlist['name']}")
        tracks = get_playlist_tracks(sp, selected_playlist['id'])

        print(f"Matching tracks with our dataset...")
        matched_tracks = pd.DataFrame()
        for chunk in load_dataset('spotify_tracks.csv'):
            matched_chunk = match_tracks_with_dataset(tracks, chunk)
            matched_tracks = pd.concat([matched_tracks, matched_chunk])

        if matched_tracks.empty:
            print("No tracks from your playlist matched our dataset. Unable to provide recommendations.")
            return

        print(f"Found {len(matched_tracks)} matches in our dataset.")

        print("\nGenerating recommendations...")
        print(f"Columns in matched_tracks: {matched_tracks.columns}")
        print(f"First few rows of matched_tracks:\n{matched_tracks.head()}")
        print(f"Shape of matched_tracks: {matched_tracks.shape}")
        print(f"Shape of all_features: {all_features.shape}")
        print(f"Shape of pca_features: {pca_features.shape}")
        recommendations = get_recommendations(matched_tracks, matched_tracks, recommender, ipca, scaler, le)

        print("\nRecommendations based on your playlist:")
        print(recommendations.to_string(index=False))

        create_playlist_choice = input(
            "\nDo you want to create a playlist with these recommendations? (yes/no): ").lower()
        if create_playlist_choice == 'yes':
            playlist_name = input("Enter a name for the new playlist: ")
            playlist_url = create_playlist(sp, user_id, playlist_name, recommendations['id'].tolist())
            print(f"\nPlaylist created successfully! You can find it here: {playlist_url}")
        else:
            print("No playlist created. You can still see the recommendations above.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()