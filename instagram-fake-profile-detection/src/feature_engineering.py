from sklearn.feature_extraction.text import TfidfVectorizer

def process_bio(X_train, X_test):
    vectorizer = TfidfVectorizer(max_features=100)
    
    # Fit TF-IDF on training bio data
    X_train_bio = vectorizer.fit_transform(X_train['bio']).toarray()
    X_test_bio = vectorizer.transform(X_test['bio']).toarray()
    
    return X_train_bio, X_test_bio, vectorizer
