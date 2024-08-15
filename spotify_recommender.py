import os
import time
import random
from spotipy.oauth2 import SpotifyOAuth
import spotipy
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import ast


def get_spotify_client():
    client_id = os.getenv('SPOTIPY_CLIENT_ID') or input("Enter your Spotify Client ID: ")
    client_secret = os.getenv('SPOTIPY_CLIENT_SECRET') or input("Enter your Spotify Client Secret: ")
    redirect_uri = os.getenv('SPOTIPY_REDIRECT_URI') or input(
        "Enter your Redirect URI (default: http://localhost:8888/callback): ") or "http://localhost:8888/callback"

    auth_manager = SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        scope="user-library-read playlist-read-private"
    )

    return spotipy.Spotify(auth_manager=auth_manager)


def rate_limited_api_call(func, *args, **kwargs):
    max_retries = 5
    base_delay = 1
    for i in range(max_retries):
        try:
            time.sleep(base_delay)
            return func(*args, **kwargs)
        except spotipy.exceptions.SpotifyException as e:
            if e.http_status == 429:
                wait_time = int(e.headers.get('Retry-After', base_delay)) + random.randint(1, 3)
                print(f"Rate limit hit. Waiting for {wait_time} seconds.")
                time.sleep(wait_time)
            else:
                print(f"Error: {e}. Retrying in {base_delay} seconds.")
                time.sleep(base_delay)
            base_delay *= 2
    raise Exception(f"Max retries reached for function {func.__name__}")


def get_user_playlists(sp, user_id):
    playlists = []
    results = rate_limited_api_call(sp.user_playlists, user_id)
    playlists.extend(results['items'])
    while results['next']:
        results = rate_limited_api_call(sp.next, results)
        playlists.extend(results['items'])
    return playlists


def get_playlist_tracks(sp, playlist_id):
    tracks = []
    results = rate_limited_api_call(sp.playlist_tracks, playlist_id)
    tracks.extend(results['items'])
    while results['next']:
        results = rate_limited_api_call(sp.next, results)
        tracks.extend(results['items'])
    return [track['track'] for track in tracks if track['track']]


def load_dataset(file_path):
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} tracks from the dataset.")
    # Convert the 'artists' column from string representation of list to actual list
    df['artists'] = df['artists'].apply(ast.literal_eval)
    return df


def match_tracks_with_dataset(tracks, dataset):
    matched_tracks = []
    for track in tracks:
        # Match based on track name and first artist
        match = dataset[
            (dataset['name'].str.lower() == track['name'].lower()) &
            (dataset['artists'].apply(lambda x: x[0].lower()) == track['artists'][0]['name'].lower())
            ]
        if not match.empty:
            matched_tracks.append(match.iloc[0])
    return pd.DataFrame(matched_tracks)


def get_recommendations(user_tracks, dataset, n=5):
    features = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
                'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

    user_features = user_tracks[features]
    dataset_features = dataset[features]

    similarities = cosine_similarity(user_features, dataset_features)
    sim_scores = similarities.mean(axis=0)

    top_indices = sim_scores.argsort()[-n:][::-1]
    recommendations = dataset.iloc[top_indices][['name', 'artists', 'album']]
    recommendations['similarity'] = sim_scores[top_indices]

    # Convert artists list to string for display
    recommendations['artists'] = recommendations['artists'].apply(lambda x: ', '.join(x))

    return recommendations


def main():
    sp = get_spotify_client()
    dataset = load_dataset('spotify_tracks.csv')  # Make sure this file exists in your directory

    try:
        user = sp.current_user()
        print(f"Successfully authenticated as {user['display_name']}")

        user_id = user['id']
        playlists = get_user_playlists(sp, user_id)

        print("\nYour playlists:")
        for i, playlist in enumerate(playlists, 1):
            print(f"{i}. {playlist['name']} (Tracks: {playlist['tracks']['total']})")

        playlist_choice = int(input("\nEnter the number of the playlist you want to use for recommendations: ")) - 1
        selected_playlist = playlists[playlist_choice]

        print(f"\nProcessing playlist: {selected_playlist['name']}")
        tracks = get_playlist_tracks(sp, selected_playlist['id'])

        print(f"Matching tracks with our dataset...")
        matched_tracks = match_tracks_with_dataset(tracks, dataset)

        if matched_tracks.empty:
            print("No tracks from your playlist matched our dataset. Unable to provide recommendations.")
            return

        print(f"Found {len(matched_tracks)} matches in our dataset.")

        print("\nGenerating recommendations...")
        recommendations = get_recommendations(matched_tracks, dataset)

        print("\nRecommendations based on your playlist:")
        print(recommendations.to_string(index=False))

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()