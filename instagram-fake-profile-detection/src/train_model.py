import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
from src.preprocess_data import preprocess_data
from src.feature_engineering import process_bio

def train_models():
    # Get preprocessed data
    X_train, X_test, y_train, y_test = preprocess_data()
    
    # Process bio features using TF-IDF
    X_train_bio, X_test_bio, vectorizer = process_bio(X_train, X_test)
    
    # Concatenate bio features with other numerical features
    X_train_combined = pd.concat([X_train.drop(columns=['bio']).reset_index(drop=True), 
                                  pd.DataFrame(X_train_bio).reset_index(drop=True)], axis=1)
    X_test_combined = pd.concat([X_test.drop(columns=['bio']).reset_index(drop=True), 
                                 pd.DataFrame(X_test_bio).reset_index(drop=True)], axis=1)
    
    # 1. Train Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_combined, y_train)
    
    # Save the trained Random Forest model
    joblib.dump(rf_model, 'model/random_forest_model.pkl')
    
    # 2. Train Decision Tree model
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train_combined, y_train)
    
    # Save the trained Decision Tree model
    joblib.dump(dt_model, 'model/decision_tree_model.pkl')
    
    # Evaluate both models on test data
    y_pred_rf = rf_model.predict(X_test_combined)
    y_pred_dt = dt_model.predict(X_test_combined)
    
    # Random Forest accuracy
    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    print(f"Random Forest Model Accuracy: {rf_accuracy * 100:.2f}%")
    
    # Decision Tree accuracy
    dt_accuracy = accuracy_score(y_test, y_pred_dt)
    print(f"Decision Tree Model Accuracy: {dt_accuracy * 100:.2f}%")
    
    return rf_model, dt_model

if __name__ == "__main__":
    train_models()
