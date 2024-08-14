# Spotify Recommendation Engine

## Import necessary libraries

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from tqdm import tqdm
import joblib

## Spotify API credentials

CLIENT_ID = 'your_client_id'
CLIENT_SECRET = 'your_client_secret'
REDIRECT_URI = 'http://localhost:8888/callback'
SCOPE = 'user-library-read playlist-read-private playlist-modify-public'

## Initialize Spotify client

def initialize_spotify_client():
    auth_manager = SpotifyOAuth(client_id=CLIENT_ID,
                                client_secret=CLIENT_SECRET,
                                redirect_uri=REDIRECT_URI,
                                scope=SCOPE)
    return spotipy.Spotify(auth_manager=auth_manager)

sp = initialize_spotify_client()

## Fetch and process Spotify tracks

def get_track_features(sp, track_id):
    try:
        features = sp.audio_features(track_id)[0]
        return {k: features[k] for k in ['danceability', 'energy', 'key', 'loudness', 'mode', 
                                         'speechiness', 'acousticness', 'instrumentalness', 
                                         'liveness', 'valence', 'tempo']}
    except:
        return None

def fetch_playlist_tracks(sp, playlist_id):
    results = sp.playlist_tracks(playlist_id)
    tracks = results['items']
    while results['next']:
        results = sp.next(results)
        tracks.extend(results['items'])
    return tracks

def process_playlist(sp, playlist_id):
    tracks = fetch_playlist_tracks(sp, playlist_id)
    track_data = []
    for item in tqdm(tracks, desc="Processing tracks"):
        track = item['track']
        features = get_track_features(sp, track['id'])
        if features:
            track_data.append({
                'id': track['id'],
                'name': track['name'],
                'artist': track['artists'][0]['name'],
                'added_at': item['added_at'],
                **features
            })
    return pd.DataFrame(track_data)

## Get user's playlists

def get_user_playlists(sp):
    playlists = sp.current_user_playlists()
    return {playlist['name']: playlist['id'] for playlist in playlists['items']}

playlists = get_user_playlists(sp)
playlists

## Process selected playlist

playlist_name = 'Your Playlist Name'  # Replace with actual playlist name
playlist_id = playlists[playlist_name]

playlist_df = process_playlist(sp, playlist_id)
playlist_df.head()

## Prepare features for recommendation

feature_cols = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 
                'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

scaler = StandardScaler()
playlist_features_scaled = scaler.fit_transform(playlist_df[feature_cols])

## Generate playlist vector

def generate_playlist_vector(features_scaled, weight_factor=1.2):
    days = (pd.to_datetime(playlist_df['added_at']).max() - pd.to_datetime(playlist_df['added_at'])).dt.days
    weights = weight_factor ** (-days)
    weighted_features = features_scaled * weights[:, np.newaxis]
    return weighted_features.mean(axis=0)

playlist_vector = generate_playlist_vector(playlist_features_scaled)

## Fetch and process a larger set of tracks for recommendations

def fetch_tracks_from_featured_playlists(sp, limit=50):
    featured_playlists = sp.featured_playlists(limit=limit)['playlists']['items']
    all_tracks = []
    for playlist in tqdm(featured_playlists, desc="Fetching playlists"):
        tracks = fetch_playlist_tracks(sp, playlist['id'])
        all_tracks.extend(tracks)
    return all_tracks

all_tracks = fetch_tracks_from_featured_playlists(sp)
all_tracks_df = pd.DataFrame([
    {**{'id': track['track']['id'], 'name': track['track']['name'], 'artist': track['track']['artists'][0]['name']},
     **get_track_features(sp, track['track']['id'])}
    for track in tqdm(all_tracks, desc="Processing tracks") if get_track_features(sp, track['track']['id'])
])

## Generate recommendations

def generate_recommendations(all_tracks_df, playlist_vector, n_recommendations=10):
    all_features_scaled = scaler.transform(all_tracks_df[feature_cols])
    similarity = cosine_similarity(all_features_scaled, playlist_vector.reshape(1, -1))
    all_tracks_df['similarity'] = similarity
    recommendations = all_tracks_df.sort_values('similarity', ascending=False).head(n_recommendations)
    return recommendations[['id', 'name', 'artist', 'similarity']]

recommendations = generate_recommendations(all_tracks_df, playlist_vector)
print(recommendations)

## Create a new playlist with recommendations

def create_recommendation_playlist(sp, recommendations, original_playlist_name):
    user_id = sp.me()['id']
    playlist_name = f"Recommendations based on {original_playlist_name}"
    new_playlist = sp.user_playlist_create(user_id, playlist_name, public=True)
    
    track_uris = [f"spotify:track:{track_id}" for track_id in recommendations['id']]
    sp.user_playlist_add_tracks(user_id, new_playlist['id'], track_uris)
    
    return new_playlist['external_urls']['spotify']

recommendation_playlist_url = create_recommendation_playlist(sp, recommendations, playlist_name)
print(f"Recommendation playlist created: {recommendation_playlist_url}")

## Save the model

joblib.dump(scaler, 'spotify_scaler.joblib')
np.save('playlist_vector.npy', playlist_vector)

print("Model saved. You can now use this to generate recommendations for new tracks!")
