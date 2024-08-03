#  Personal Movie Recommendation with LGBM Ranker

This Jupyter Notebook demonstrates how to build a movie recommendation system using the LGBM Ranker algorithm. The system provides personalized recommendations based on individual user preferences.

## Steps:

1. **Data Import:**
   - Import necessary libraries (NumPy, Pandas, scikit-learn, LightGBM, Matplotlib).
   - Load movie ratings and movie information from CSV files.
   - Preprocess data:
      - Shuffle and select a subset of ratings for efficiency.
      - Create a mapping of movie IDs to titles.
      - Merge rating and movie dataframes.

2. **Data Exploration:**
   - Visualize the distribution of ratings using histograms and scatter plots.
   - Analyze the relationship between user IDs and ratings.
   - Examine the distribution of movies rated.

3. **Feature Engineering:**
   - Define feature columns (`userId`, `movieId`) and the target variable (`rating`).
   - Specify the group column (`userId`) for personalized recommendations.

4. **Data Splitting:**
   - Split the data into training and testing sets.

5. **Model Training:**
   - Instantiate an LGBMRanker model with appropriate parameters (objective, metric, boosting type, etc.).
   - Calculate session lengths for the training set.
   - Fit the model to the training data, considering session lengths for grouping.

6. **Recommendation Generation:**
   - Create a test set with combinations of user IDs and movie IDs.
   - Predict scores for the test set using the trained model.
   - Concatenate the test set with predicted scores.

7. **Result Analysis:**
   - For selected movies, display the top recommended users based on predicted scores.
   - For selected users, display the top recommended movies based on predicted scores.
   - Visualize the distribution of predicted scores.
   - Plot the top 10 rated movies and their average scores.

## Conclusion:

The LGBM Ranker model successfully generates personalized movie recommendations, demonstrating its ability to capture individual user preferences.
