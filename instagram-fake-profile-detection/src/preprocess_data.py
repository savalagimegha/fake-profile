import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data():
    # Load the dataset
    data = pd.read_csv('data/instagram_data.csv')
    
    # Fill missing values
    data.fillna("", inplace=True)
    
    # Feature Engineering (Followers/Following Ratio)
    data['followers_to_following'] = data['followers'] / (data['following'] + 1)  # Add 1 to avoid division by zero
    
    # Split into features and labels
    X = data[['followers', 'following', 'posts', 'followers_to_following', 'bio']]
    y = data['is_fake']
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Return preprocessed data
    return X_train, X_test, y_train, y_test
