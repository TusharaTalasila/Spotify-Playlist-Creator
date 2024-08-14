## Import necessary libraries

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from datetime import datetime

## Spotify API credentials

CLIENT_ID = 'your_client_id' //removed for security
CLIENT_SECRET = 'your_client_secret' //removed for security
REDIRECT_URI = 'http://localhost:8888/callback'
SCOPE = 'user-library-read playlist-read-private'

## Load and preprocess the Spotify dataset

def load_and_preprocess_data(file_path): 
    spotify_data = pd.read_csv(file_path)
    
    # One-hot encoding for genre and key
    genre_ohe = pd.get_dummies(spotify_data.genre, prefix='genre')
    key_ohe = pd.get_dummies(spotify_data.key, prefix='key')
    
    # Select features for scaling
    features_to_scale = ['acousticness', 'danceability', 'duration_ms', 'energy', 
                         'instrumentalness', 'liveness', 'loudness', 'speechiness', 
                         'tempo', 'valence']
    
    # Scale selected features
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(spotify_data[features_to_scale])
    
    # Create a new dataframe with scaled features
    spotify_features_df = pd.DataFrame(scaled_features, columns=features_to_scale)
    
    # Add track_id and one-hot encoded columns
    spotify_features_df['track_id'] = spotify_data['track_id']
    spotify_features_df = pd.concat([spotify_features_df, genre_ohe, key_ohe], axis=1)
    
    return spotify_features_df

spotify_features_df = load_and_preprocess_data('path_to_your_spotify_dataset.csv')
spotify_features_df.head()

## Initialize Spotify client

def initialize_spotify_client():
    auth_manager = SpotifyOAuth(client_id=CLIENT_ID,
                                client_secret=CLIENT_SECRET,
                                redirect_uri=REDIRECT_URI,
                                scope=SCOPE)
    return spotipy.Spotify(auth_manager=auth_manager)

sp = initialize_spotify_client()

## Get user's playlists

def get_user_playlists(sp):
    playlists = sp.current_user_playlists()
    return {playlist['name']: playlist['id'] for playlist in playlists['items']}

playlists = get_user_playlists(sp)
playlists

## Generate playlist dataframe

def generate_playlist_df(sp, playlist_id, spotify_data):
    playlist_tracks = sp.playlist_tracks(playlist_id)
    playlist = []
    
    for item in playlist_tracks['items']:
        track = item['track']
        playlist.append({
            'artist': track['artists'][0]['name'],
            'track_name': track['name'],
            'track_id': track['id'],
            'date_added': item['added_at']
        })
    
    playlist_df = pd.DataFrame(playlist)
    playlist_df['date_added'] = pd.to_datetime(playlist_df['date_added'])
    playlist_df = playlist_df[playlist_df['track_id'].isin(spotify_data['track_id'])]
    return playlist_df.sort_values('date_added', ascending=False)

# Select a playlist
playlist_name = 'Your Playlist Name'  # Replace with actual playlist name
playlist_id = playlists[playlist_name]

playlist_df = generate_playlist_df(sp, playlist_id, spotify_features_df)
playlist_df.head()

## Generate playlist vector

def generate_playlist_vector(spotify_features_df, playlist_df, weight_factor):
    playlist_features = spotify_features_df[spotify_features_df['track_id'].isin(playlist_df['track_id'])]
    playlist_features = playlist_features.merge(playlist_df[['track_id', 'date_added']], on='track_id', how='inner')
    
    most_recent_date = playlist_features['date_added'].max()
    playlist_features['days_from_recent'] = (most_recent_date - playlist_features['date_added']).dt.days
    playlist_features['weight'] = weight_factor ** (-playlist_features['days_from_recent'])
    
    weighted_features = playlist_features.drop(['track_id', 'date_added', 'days_from_recent', 'weight'], axis=1).mul(playlist_features['weight'], axis=0)
    playlist_vector = weighted_features.sum()
    
    return playlist_vector

playlist_vector = generate_playlist_vector(spotify_features_df, playlist_df, weight_factor=1.2)
playlist_vector

## Generate recommendations

def generate_recommendations(spotify_features_df, playlist_vector, n_recommendations=10):
    # Calculate similarity
    similarity = cosine_similarity(spotify_features_df.drop('track_id', axis=1), playlist_vector.values.reshape(1, -1))
    
    # Add similarity scores to the dataframe
    spotify_features_df['similarity'] = similarity
    
    # Sort by similarity and get top recommendations
    recommendations = spotify_features_df.sort_values('similarity', ascending=False).head(n_recommendations)
    
    return recommendations[['track_id', 'similarity']]

recommendations = generate_recommendations(spotify_features_df, playlist_vector)
recommendations

## Print recommendations

print("Recommended tracks:")
for _, row in recommendations.iterrows():
    track = sp.track(row['track_id'])
    print(f"{track['name']} by {track['artists'][0]['name']} (Similarity: {row['similarity']:.4f})")
