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

def get_spotify_client():
    client_id = os.getenv('SPOTIPY_CLIENT_ID') or input("Enter your Spotify Client ID: ")
    client_secret = os.getenv('SPOTIPY_CLIENT_SECRET') or input("Enter your Spotify Client Secret: ")
    redirect_uri = os.getenv('SPOTIPY_REDIRECT_URI') or input(
        "Enter your Redirect URI (default: http://localhost:8888/callback): ") or "http://localhost:8888/callback"

    auth_manager = SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        scope="user-library-read playlist-read-private playlist-modify-public playlist-modify-private"
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


def create_playlist(sp, user_id, playlist_name, track_ids):
    playlist = rate_limited_api_call(sp.user_playlist_create, user_id, playlist_name)
    rate_limited_api_call(sp.playlist_add_items, playlist['id'], track_ids)
    return playlist['external_urls']['spotify']