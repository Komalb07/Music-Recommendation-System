# Music-Recommendation-System

Our project aimed to develop a personalized music recommendation system using data from Spotify. There are three main objectives based on various audio features:
1) Predict the popularity of a song
2) Classify songs into "Popular" and "Not Popular" categories
3) Recommend top 5 songs for a specific user based on the songs in their playlists. 

To build the system, we collected a comprehensive dataset using the Spotify API, which included attributes such as danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, mode, key, and popularity. Our dataset comprised 5,333 songs, each with detailed audio features and associated popularity scores.

We conducted Exploratory Data Analysis (EDA) to understand the distribution and relationships of these features. Through univariate and bivariate analyses, we uncovered key insights, such as the positive correlations between danceability and energy.

During the data preparation and feature selection phase, we handled missing values by dropping rows with null values. We used the Ordinary Least Squares (OLS) method to select significant features for our modeling phase. The features identified as significant were danceability, energy, loudness, acousticness, instrumentalness, liveness, and valence.

For the modeling approach, we employed both regression and classification models. In the regression models, Linear Regression and Ridge Regression each provided an R² value of 0.062. Polynomial Regression improved the R² value to 0.101, while the Random Forest Regressor achieved the best performance with an R² value of 0.269. For classification models, Logistic Regression achieved an accuracy of 0.604, Support Vector Machine (SVM) improved the accuracy to 0.630, and the Random Forest Classifier provided the highest accuracy of 0.671.

The final component of our project was the development of a recommendation system using a content-based filtering approach. We utilized cosine similarity to measure the similarity between the audio features of different songs and generate personalized recommendations based on user listening history.
