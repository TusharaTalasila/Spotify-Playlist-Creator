# Spotify Playlist Recommendation Engine

## Overview
This project implements a hybrid recommendation system using the Spotify API and a pre-existing dataset of tracks. It analyzes a user's playlist and generates personalized music recommendations based on audio features and track metadata.

## Features
- Authenticates with Spotify and fetches user's playlists
- Allows users to select a playlist for analysis
- Matches user's tracks with a large pre-existing dataset
- Implements a content-based filtering system using cosine similarity
- Generates recommendations based on audio features
- Works with public playlists, respecting Spotify's privacy policies

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
4. Ensure you have the `spotify_tracks.csv` file in your project directory

## Usage
1. Run the script:
   ```
   python enginecode.py
   ```
2. Enter your Spotify API credentials when prompted
3. Select a playlist from your Spotify account
4. The script will analyze the playlist and generate recommendations


## Notes
- The quality of recommendations depends on how well the user's tracks match with the pre-existing dataset
- Only works with tracks that exist in the provided dataset (Modification had to be made to Spotify API issues)
