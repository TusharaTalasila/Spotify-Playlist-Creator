# Spotify Playlist Recommendation Engine

## Overview
This project implements an advanced recommendation system using the Spotify API and a comprehensive dataset of over 1 million tracks. It analyzes a user's playlist and generates personalized music recommendations using machine learning techniques.

## Features
- Authenticates with Spotify and fetches user's playlists
- Generates recommendations using Incremental PCA and Nearest Neighbors algorithm
- Processes large-scale music data efficiently
- Creates a new Spotify playlist with recommended tracks

## Technologies Used
- Python
- Pandas for data manipulation
- NumPy for numerical computations
- Scikit-learn for machine learning algorithms
- Spotipy for Spotify API integration

## Setup and Installation
1. Create a python project
2. Copy .py files into project bin
3. Install required packages:
   ```
   pip install spotipy pandas scikit-learn numpy
   ```
4. Set up your Spotify Developer account and obtain API credentials
5. Ensure you have the `spotify_tracks.csv` file in your project directory (https://www.kaggle.com/code/muhammedtausif/top-songs-eda/input)
6. Update your Spotify Developer App to include the scopes: `playlist-modify-public` and `playlist-modify-private`

## Usage
1. Run the script:
   ```
   python main.py
   ```
2. Follow the prompts to select a playlist and generate recommendations

## How It Works
1. Authenticates with Spotify
2. Loads and preprocesses the track dataset
3. Applies Incremental PCA for dimensionality reduction
4. Builds a Nearest Neighbors model for similarity computation
5. Fetches and analyzes the user's selected playlist
6. Generates recommendations using the trained model
7. Creates a new playlist with recommendations if requested
