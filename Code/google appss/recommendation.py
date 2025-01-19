from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Load and preprocess the data
def preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # Normalize numerical features
    numerical_features = ['Rating', 'Sentiment_Polarity', 'Discrepancy', 'User_Engagement']
    scaler = MinMaxScaler()

    df[numerical_features] = df[numerical_features].fillna(0)
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    # Handle missing categorical values
    df['Category'] = df['Category'].fillna('Unknown')
    df['Success Category'] = df['Success Category'].fillna('Unknown')

    # Apply Feature Weighting
    df['Weighted_Rating'] = df['Rating'] * 0.5
    df['Weighted_Sentiment_Polarity'] = df['Sentiment_Polarity'] * 0.3
    df['Weighted_Reviews_Installs'] = df['User_Engagement'] * 0.1
    df['Weighted_Discrepancy'] = df['Discrepancy'] * -0.2

    # Combine features
    features = ['Category', 'Success Category', 'Weighted_Rating', 'Weighted_Sentiment_Polarity',
                'Weighted_Reviews_Installs', 'Weighted_Discrepancy']
    feature_matrix = pd.get_dummies(df[features])

    # Compute similarity
    similarity_matrix = cosine_similarity(feature_matrix)

    # Map indices
    df_to_feature_matrix_indices = {index: idx for idx, index in enumerate(df.index)}

    return df, similarity_matrix, df_to_feature_matrix_indices


def content_based_recommendations_by_category(df, similarity_matrix, df_to_feature_matrix_indices, category, n_recommendations=5, discrepancy_threshold=0.2):
    # Step 1: Filter apps in the specified category
    category_apps = df[(df['Category'] == category) &
                       (df['Discrepancy'] <= discrepancy_threshold)]
    if category_apps.empty:
        return f"No apps found in the category '{category}' with discrepancy <= {discrepancy_threshold}."

    # Step 2: Map the indices of category apps to the feature matrix
    category_indices = [df_to_feature_matrix_indices[idx] for idx in category_apps.index]

    # Step 3: Compute mean similarity scores for apps in the category
    scores = similarity_matrix[category_indices].mean(axis=0)

    # Step 4: Sort apps by similarity score
    top_indices = scores.argsort()[::-1][:n_recommendations]  # Indices of top scores
    top_apps = df.iloc[top_indices][['App', 'Category', 'Discrepancy']]  # Relevant columns

    # Update the index to start from 1
    top_apps.index = range(1, len(top_apps) + 1)

    return top_apps
