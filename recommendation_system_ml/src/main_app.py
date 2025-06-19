import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix

# Suppress warnings
warnings.filterwarnings('ignore')

# =====================
# DATA LOADING & SETUP
# =====================
def load_data(file_path):
    """Load and preprocess the ratings dataset"""
    # Load dataset without headers
    df = pd.read_csv(file_path, header=None)
    
    # Add column names and drop timestamp
    df.columns = ['user_id', 'prod_id', 'rating', 'timestamp']
    df = df.drop('timestamp', axis=1)
    
    # Create a deep copy for backup
    df_copy = df.copy(deep=True)
    return df, df_copy

# =====================
# EXPLORATORY DATA ANALYSIS
# =====================
def perform_eda(df):
    """Perform exploratory data analysis on the dataset"""
    print("\n=== Exploratory Data Analysis ===")
    
    # Dataset shape
    rows, columns = df.shape
    print(f"No of rows = {rows}")
    print(f"No of columns = {columns}")
    
    # Data types
    print("\nData types:")
    print(df.info())
    
    # Missing values
    print("\nMissing values:")
    print(df.isna().sum())
    
    # Summary statistics
    print("\nRating summary statistics:")
    print(df['rating'].describe())
    
    # Rating distribution plot
    plt.figure(figsize=(12, 6))
    df['rating'].value_counts(normalize=True).plot(kind='bar')
    plt.title('Rating Distribution')
    plt.xlabel('Rating')
    plt.ylabel('Percentage')
    plt.show()
    
    # Unique users and products
    print(f"\nNumber of unique USERS = {df['user_id'].nunique()}")
    print(f"Number of unique ITEMS = {df['prod_id'].nunique()}")
    
    # Top 10 users by number of ratings
    print("\nTop 10 users by number of ratings:")
    most_rated = df.groupby('user_id').size().sort_values(ascending=False)[:10]
    print(most_rated)

# =====================
# DATA PREPROCESSING
# =====================
def preprocess_data(df, min_ratings=50):
    """Filter users with minimum ratings and create interaction matrix"""
    print("\n=== Data Preprocessing ===")
    
    # Filter users with at least min_ratings
    counts = df['user_id'].value_counts()
    df_final = df[df['user_id'].isin(counts[counts >= min_ratings].index)]
    
    print(f"Final dataset size: {len(df_final)}")
    print(f"Unique users in final data: {df_final['user_id'].nunique()}")
    print(f"Unique products in final data: {df_final['prod_id'].nunique()}")
    
    # Create interaction matrix
    final_ratings_matrix = df_final.pivot(
        index='user_id', 
        columns='prod_id', 
        values='rating'
    ).fillna(0)
    
    # Calculate matrix density
    given_ratings = np.count_nonzero(final_ratings_matrix)
    possible_ratings = final_ratings_matrix.shape[0] * final_ratings_matrix.shape[1]
    density = (given_ratings / possible_ratings) * 100
    print(f"Density: {density:.2f}%")
    
    return df_final, final_ratings_matrix

# =====================
# RANK-BASED RECOMMENDATIONS
# =====================
def rank_based_recommendations(df_final):
    """Generate rank-based product recommendations"""
    print("\n=== Rank-Based Recommendations ===")
    
    # Calculate average ratings and rating counts
    avg_rating = df_final.groupby('prod_id').mean()['rating']
    count_rating = df_final.groupby('prod_id').count()['rating']
    final_rating = pd.DataFrame({
        'avg_rating': avg_rating,
        'rating_count': count_rating
    }).sort_values('avg_rating', ascending=False)
    
    # Recommendation function
    def top_n_products(final_rating, n, min_interaction):
        recommendations = final_rating[final_rating['rating_count'] > min_interaction]
        return recommendations.sort_values('avg_rating', ascending=False).index[:n]
    
    # Get recommendations with different thresholds
    print("\nTop 5 products with at least 50 interactions:")
    print(list(top_n_products(final_rating, 5, 50)))
    
    print("\nTop 5 products with at least 100 interactions:")
    print(list(top_n_products(final_rating, 5, 100)))
    
    return final_rating

# =====================
# COLLABORATIVE FILTERING
# =====================
def collaborative_filtering(final_ratings_matrix):
    """User-based collaborative filtering recommendations"""
    print("\n=== User-Based Collaborative Filtering ===")
    
    # Add numeric index
    final_ratings_matrix = final_ratings_matrix.copy()
    final_ratings_matrix['user_index'] = np.arange(0, final_ratings_matrix.shape[0])
    final_ratings_matrix.set_index('user_index', inplace=True)
    
    # Similar users function
    def similar_users(user_index, interactions_matrix):
        similarity = []
        for user in range(interactions_matrix.shape[0]):
            sim = cosine_similarity(
                [interactions_matrix.loc[user_index]],
                [interactions_matrix.loc[user]]
            )
            similarity.append((user, sim))
        
        similarity.sort(key=lambda x: x[1], reverse=True)
        most_similar_users = [tup[0] for tup in similarity]
        similarity_score = [tup[1] for tup in similarity]
        
        # Remove the user themselves
        most_similar_users.remove(user_index)
        similarity_score.pop(0)
        
        return most_similar_users, similarity_score
    
    # Recommendation function
    def recommendations(user_index, num_products, interactions_matrix):
        most_similar_users = similar_users(user_index, interactions_matrix)[0]
        prod_ids = set(interactions_matrix.columns[
            np.where(interactions_matrix.loc[user_index] > 0)
        ])
        
        recs = []
        observed = prod_ids.copy()
        
        for user in most_similar_users:
            if len(recs) < num_products:
                similar_prods = set(interactions_matrix.columns[
                    np.where(interactions_matrix.loc[user] > 0)
                ])
                new_recs = list(similar_prods.difference(observed))
                recs.extend(new_recs)
                observed = observed.union(similar_prods)
            else:
                break
        
        return recs[:num_products]
    
    # Generate recommendations for sample users
    print("\nTop 5 recommendations for user index 3:")
    print(recommendations(3, 5, final_ratings_matrix))
    
    print("\nTop 5 recommendations for user index 1521:")
    print(recommendations(1521, 5, final_ratings_matrix))
    
    return final_ratings_matrix

# =====================
# MODEL-BASED RECOMMENDATIONS (SVD)
# =====================
def svd_recommendations(final_ratings_matrix):
    """SVD-based recommendation system"""
    print("\n=== SVD-Based Recommendations ===")
    
    # Convert to sparse matrix
    final_ratings_sparse = csr_matrix(final_ratings_matrix.values)
    
    # Perform SVD
    U, s, Vt = svds(final_ratings_sparse, k=50)
    sigma = np.diag(s)
    
    # Reconstruct predicted ratings matrix
    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)
    preds_df = pd.DataFrame(
        abs(all_user_predicted_ratings),
        columns=final_ratings_matrix.columns
    )
    preds_matrix = csr_matrix(preds_df.values)
    
    # Recommendation function
    def recommend_items(user_index, interactions_matrix, preds_matrix, num_recs):
        user_ratings = interactions_matrix[user_index, :].toarray().reshape(-1)
        user_predictions = preds_matrix[user_index, :].toarray().reshape(-1)
        
        temp = pd.DataFrame({
            'user_ratings': user_ratings,
            'user_predictions': user_predictions
        })
        temp['Recommended Products'] = np.arange(len(user_ratings))
        temp = temp.set_index('Recommended Products')
        
        # Filter unrated items
        temp = temp.loc[temp.user_ratings == 0]
        temp = temp.sort_values('user_predictions', ascending=False)
        
        print(f"\nTop {num_recs} recommendations for user {user_index}:")
        print(temp['user_predictions'].head(num_recs))
    
    # Generate recommendations
    recommend_items(121, final_ratings_sparse, preds_matrix, 5)
    recommend_items(100, final_ratings_sparse, preds_matrix, 10)
    
    # Evaluation
    avg_actual = final_ratings_matrix.mean()
    avg_preds = preds_df.mean()
    
    rmse_df = pd.DataFrame({
        'Avg_actual_ratings': avg_actual,
        'Avg_predicted_ratings': avg_preds
    })
    
    print("\nEvaluation Metrics (Sample):")
    print(rmse_df.head())
    
    return rmse_df

# =====================
# MAIN EXECUTION
# =====================
if __name__ == "__main__":
    # File path - update this according to your environment
    FILE_PATH = '/content/drive/MyDrive/ratings_Electronics.csv'
    
    # Load data
    df, df_copy = load_data(FILE_PATH)
    
    # Exploratory Data Analysis
    perform_eda(df)
    
    # Preprocessing (filter active users)
    df_final, final_ratings_matrix = preprocess_data(df)
    
    # Rank-based recommendations
    final_rating = rank_based_recommendations(df_final)
    
    # Collaborative filtering
    final_ratings_matrix = collaborative_filtering(final_ratings_matrix)
    
    # SVD recommendations
    rmse_df = svd_recommendations(final_ratings_matrix)
