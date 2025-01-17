# Music Recommendation System

This repository contains the implementation of a **Music Recommendation System** designed to predict song popularity and provide personalized recommendations based on user preferences. The project leverages machine learning and audio feature analysis to enhance the user experience for music streaming platforms.

## Project Overview

With the exponential growth of music streaming services, it has become essential to offer personalized song recommendations to users. This project addresses this need by:
- Predicting song popularity using regression models.
- Categorizing songs as "Popular" or "Not Popular" using classification models.
- Providing personalized recommendations using content-based filtering.

Our system is built on a dataset extracted from Spotify, enriched with audio features like danceability, energy, loudness, and more.

## Key Features

1. **Audio Feature Analysis**:
   - Extracted attributes such as danceability, energy, acousticness, and valence using the Spotify API.
   - Performed Exploratory Data Analysis (EDA) to uncover insights into feature distributions and relationships.

2. **Popularity Prediction**:
   - Implemented regression models including:
     - Linear Regression
     - Ridge Regression
     - Random Forest Regressor (achieved R² = 0.269)

3. **Classification**:
   - Categorized songs into "Popular" or "Not Popular" using:
     - Logistic Regression
     - Support Vector Machine (SVM)
     - Random Forest Classifier (achieved accuracy = 0.671)

4. **Recommendation System**:
   - Built a content-based recommendation engine using cosine similarity.
   - Generated personalized song recommendations based on user listening history.

## Methodology

### 1. Data Collection
- Extracted data for 5,333 songs using the Spotify API and the `spotipy` library.
- Key attributes included audio features, popularity scores, and metadata.

### 2. Feature Engineering
- Selected significant features like danceability, energy, and valence using Ordinary Least Squares (OLS).
- Normalized features for consistent model performance.

### 3. Modeling
- **Regression Models**: Predicted song popularity scores.
- **Classification Models**: Identified popular vs. non-popular songs.
- **Content-Based Recommendation Engine**: Recommended songs using cosine similarity.

### 4. Evaluation
- Used R², accuracy, recall, and precision metrics to assess model performance.

## Results

### Regression Models
| Model                  | R² Score |
|------------------------|----------|
| Linear Regression      | 0.062    |
| Ridge Regression       | 0.062    |
| Polynomial Regression  | 0.101    |
| Random Forest Regressor| 0.269    |

### Classification Models
| Model                  | Accuracy | Recall | Precision |
|------------------------|----------|--------|-----------|
| Logistic Regression    | 0.604    | 0.780  | 0.592     |
| SVM                    | 0.630    | 0.778  | 0.616     |
| Random Forest Classifier| 0.671   | 0.762  | 0.661     |


## Future Enhancements

- **Feature Expansion**: Incorporate user demographics and social listening habits.
- **Real-Time Recommendations**: Enable dynamic updates based on recent user activity.
- **Scalability**: Optimize for large datasets and high user volumes.
