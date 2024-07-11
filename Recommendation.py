import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Load the data
df = pd.read_excel('/Users/komalb/MS/Math 448/Project/Final Songs.xlsx')
df.dropna(inplace=True)

# List of features to scale
features_to_scale = ['Danceability', 'Energy', 'Loudness', 'Speechiness', 'Acousticness', 'Instrumentalness',
                     'Liveness', 'Valence', 'Tempo']

# Initialize the scaler
scaler = StandardScaler()

# Fit and transform the features, ensuring the indices are preserved
features_scaled_df = pd.DataFrame(scaler.fit_transform(df[features_to_scale]), columns=features_to_scale, index=df.index)

# Replace the original features with the scaled values
df[features_to_scale] = features_scaled_df

# Reset the index of the DataFrame to ensure continuous and aligned indices
df.reset_index(drop=True, inplace=True)

# List of significant features
significant_features = ['Danceability', 'Energy', 'Loudness', 'Acousticness', 'Instrumentalness', 'Liveness', 'Valence']

def recommend_songs(user_id, df, significant_features, num_recommendations=5):
    # Filter songs that are in the user's playlist
    user_songs = df[df['User id'] == user_id]
    user_playlist_songs = user_songs['Song id'].unique()

    # Ensure only songs that are in the dataset are considered
    valid_user_playlist_songs = df[df['Song id'].isin(user_playlist_songs)]
    user_song_indices = valid_user_playlist_songs.index

    if len(user_song_indices) == 0:
        print("No valid songs found in the user's playlist.")
        return pd.DataFrame()

    # Calculate cosine similarity matrix
    similarity_matrix = cosine_similarity(df[significant_features])

    # Calculate mean similarity score for each song with the user's playlist
    mean_similarity_scores = similarity_matrix[user_song_indices].mean(axis=0)

    # Create a DataFrame for similarity scores
    similarity_scores_df = pd.DataFrame({
        'Song id': df['Song id'],
        'Similarity Score': mean_similarity_scores
    })

    # Remove songs that are already in the user's playlist
    similarity_scores_df = similarity_scores_df[~similarity_scores_df['Song id'].isin(user_playlist_songs)]

    # Sort by similarity score and get top recommendations
    top_recommendations = similarity_scores_df.sort_values(by='Similarity Score', ascending=False).head(
        num_recommendations)

    # Merge with the original DataFrame to get song details
    recommended_songs = top_recommendations.merge(df[['Song id', 'Song Name', 'Artist(s)']],
                                                  on='Song id').drop_duplicates(subset='Song id')
    return recommended_songs

user_id = '31i4snwfisknrnd64ta3njgs6v2e'
recommended_songs = recommend_songs(user_id, df, significant_features)

print("Recommended songs for user: ", user_id)
print(recommended_songs[['Song Name', 'Similarity Score']].to_string(index=False))