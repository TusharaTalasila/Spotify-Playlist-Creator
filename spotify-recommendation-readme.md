# Spotify Playlist Recommendation Engine

## Overview
This project implements a machine learning-based song recommendation system using the Spotify API. It analyzes a user's playlist to generate personalized music recommendations based on audio features and listening history.

## Features
- Fetches and processes track data directly from Spotify
- Implements a content-based filtering system using cosine similarity
- Applies time decay weighting to favor recently added tracks
- Creates a new Spotify playlist with recommended tracks
- Saves the trained model for future use

## Technologies Used
- Python
- Pandas for data manipulation
- Scikit-learn for machine learning algorithms
- Spotipy for Spotify API integration
- Joblib for model persistence

## Setup and Installation
1. Clone the repository:
   ```
   git clone https://github.com/YourUsername/Spotify-Playlist-Recommender.git
   ```
2. Install required packages:
   ```
   pip install -r requirements.txt
   ```
3. Set up your Spotify Developer account and obtain API credentials
4. Update the `CLIENT_ID`, `CLIENT_SECRET`, and `REDIRECT_URI` in the notebook

## Usage
1. Open the Jupyter Notebook `Spotify_Recommendation_Engine.ipynb`
2. Run the cells in order
3. When prompted, select the playlist you want to base recommendations on
4. The script will generate recommendations and create a new playlist in your Spotify account

## Future Improvements
- Implement a user interface for easier interaction
- Incorporate collaborative filtering for more diverse recommendations
- Optimize the algorithm for larger datasets and faster processing

## Contributing
Contributions, issues, and feature requests are welcome!

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
