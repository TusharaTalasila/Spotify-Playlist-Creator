# Spotify Playlist Recommendation Engine

## Overview
This project implements a hybrid recommendation system using the Spotify API and a pre-existing dataset of tracks. It analyzes a user's playlist, generates personalized music recommendations based on audio features and track metadata, and offers the option to create a new Spotify playlist with these recommendations.

## Features
- Authenticates with Spotify and fetches user's playlists
- Allows users to select a playlist for analysis
- Generates recommendations based on audio features
- Allows users to create a new Spotify playlist with recommended tracks

## Technologies Used
- Python
- Pandas for data manipulation
- Scikit-learn for machine learning algorithms (cosine similarity)
- Spotipy for Spotify API integration

## Setup and Installation
1. Clone the repository
2. Install required packages:
   ```
   pip install spotipy pandas scikit-learn
   ```
3. Set up your Spotify Developer account and obtain API credentials
4. Ensure you have the `spotify_tracks.csv` file in your project directory download from: https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs?resource=download
5. Update your Spotify Developer App to include the scopes: `playlist-modify-public` and `playlist-modify-private`

## Usage
1. Run the script:
   ```
   python spotify_recommender.py
   ```
2. Follow the prompts

## How It Works
1. Authenticates with Spotify
2. Fetches user's playlists
3. User selects a playlist
4. Retrieves tracks from the selected playlist
5. Matches these tracks with the pre-existing dataset
6. Calculates similarity between playlist tracks and dataset tracks
7. Generates and displays recommendations
8. Offers to create a new playlist with recommendations
9. If chosen, creates the playlist and adds recommended tracks

## Notes
- The quality of recommendations depends on how well the user's tracks match with the pre-existing dataset
- Only works with tracks that exist in the provided dataset
