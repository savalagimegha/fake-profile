import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from src.preprocess_data import preprocess_data
from src.feature_engineering import process_bio

def evaluate_models():
    # Load trained models
    rf_model = joblib.load('model/random_forest_model.pkl')
    dt_model = joblib.load('model/decision_tree_model.pkl')
    
    # Get preprocessed data
    X_train, X_test, y_train, y_test = preprocess_data()
    
    # Process bio features using the same vectorizer
    X_train_bio, X_test_bio, _ = process_bio(X_train, X_test)
    
    # Concatenate bio features with other numerical features
    X_test_combined = pd.concat([X_test.drop(columns=['bio']).reset_index(drop=True), 
                                 pd.DataFrame(X_test_bio).reset_index(drop=True)], axis=1)
    
    # Predict using both models
    y_pred_rf = rf_model.predict(X_test_combined)
    y_pred_dt = dt_model.predict(X_test_combined)
    
    # Random Forest Evaluation
    print("Random Forest Classification Report:")
    print(classification_report(y_test, y_pred_rf))
    print("Random Forest Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_rf))
    
    # Decision Tree Evaluation
    print("Decision Tree Classification Report:")
    print(classification_report(y_test, y_pred_dt))
    print("Decision Tree Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_dt))

if __name__ == "__main__":
    evaluate_models()
