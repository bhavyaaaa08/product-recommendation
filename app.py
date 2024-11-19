import streamlit as st
import pandas as pd
import os
import kagglehub
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error

# Load and preprocess dataset
@st.cache_data
def load_and_preprocess_data():

    # Load dataset (you can replace this with your dataset path)
    path = kagglehub.dataset_download("vibivij/amazon-electronics-rating-datasetrecommendation")
    file_name = "ratings_Electronics.csv"
    df = pd.read_csv(os.path.join(path, file_name))
    df.columns = ['user_id', 'prod_id', 'rating', 'timestamp'] #Adding column names
    df = df.drop('timestamp', axis=1)
    
    # Filter users with at least 50 ratings
    counts = df['user_id'].value_counts()
    df_final = df[df['user_id'].isin(counts[counts >= 50].index)]
    
    return df_final

# Prepare interaction matrix
def prepare_interaction_matrix(df):
    return df.pivot_table(index='user_id', columns='prod_id', values='rating', fill_value=0)

# Rank-Based Recommendations
def rank_based_recommendations(df, n=5, min_interactions=50):
    avg_rating = df.groupby('prod_id')['rating'].mean()
    count_rating = df.groupby('prod_id')['rating'].count()
    final_rating = pd.DataFrame({'avg_rating': avg_rating, 'rating_count': count_rating})
    
    recommendations = final_rating[final_rating['rating_count'] > min_interactions]
    return recommendations.sort_values('avg_rating', ascending=False).head(n).index.tolist()

# User-Based Collaborative Filtering
def user_based_recommendations(user_id, df, num_recommendations=5):
    interaction_matrix = prepare_interaction_matrix(df)
    
    if user_id not in interaction_matrix.index:
        st.warning(f"User {user_id} not found in the dataset.")
        return []
    
    # Compute cosine similarity
    user_similarities = cosine_similarity(interaction_matrix.loc[user_id].values.reshape(1, -1), 
                                          interaction_matrix.values)
    
    similar_users_indices = user_similarities[0].argsort()[::-1][1:11]  # Top 10 similar users
    similar_users = interaction_matrix.index[similar_users_indices]
    
    # Find unrated products
    user_rated_products = set(interaction_matrix.columns[interaction_matrix.loc[user_id] > 0])
    recommendations = []
    
    for similar_user in similar_users:
        similar_user_products = set(interaction_matrix.columns[interaction_matrix.loc[similar_user] > 0])
        new_recommendations = list(similar_user_products - user_rated_products)
        recommendations.extend(new_recommendations)
        
        if len(recommendations) >= num_recommendations:
            break
    
    return recommendations[:num_recommendations]

# Streamlit App
def main():
    st.title("Product Recommendation System")
    
    # Load data
    df = load_and_preprocess_data()
    
    # Sidebar navigation
    recommendation_type = st.sidebar.radio(
        "Select Recommendation Type", 
        ["Rank-Based", "User-Based"]
    )
    
    if recommendation_type == "Rank-Based":
        st.header("Top Rated Products")
        min_interactions = st.slider("Minimum Interactions", 10, 100, 50)
        top_products = rank_based_recommendations(df, min_interactions=min_interactions)
        st.write("Top Products:", top_products)
    
    else:
        st.header(f"{recommendation_type} Recommendations")
        user_id = st.text_input("Enter User ID", value="")
        
        if user_id:
            try:
                if recommendation_type == "User-Based":
                    recommendations = user_based_recommendations(user_id, df)
                
                if recommendations:
                    st.subheader("Recommended Product IDs:")
                    st.write(recommendations)
                else:
                    st.warning("No recommendations found.")
            
            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()