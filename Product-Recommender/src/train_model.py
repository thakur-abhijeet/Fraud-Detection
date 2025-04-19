import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

def load_data(file_path):
    """Load dataset from a CSV file"""
    return pd.read_csv(file_path)

def clean_data(df):
    """Clean dataset by handling missing values and duplicates"""
    df = df.copy()
    
    # Handle missing values
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna('Unknown')

    # Drop duplicates
    df.drop_duplicates(inplace=True)
    
    return df

def prepare_data(df):
    """Prepare data for modeling"""
    df = df.copy()
    
    # Separate features and target
    X = df.drop('PurchaseIntent', axis=1)
    y = df['PurchaseIntent']
    
    # Convert categorical variables to dummy variables
    X = pd.get_dummies(X, drop_first=True)
    
    return X, y

def build_and_train_model(X_train, y_train):
    """Build and train a Random Forest model"""
    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    
    base_model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=5,
        n_jobs=-1,
        scoring='accuracy'
    )
    
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def main():
    # Load and prepare the data
    print("Loading data...")
    df = load_data('../data/consumerdata.csv')
    
    print("Cleaning data...")
    df = clean_data(df)
    
    print("Preparing data...")
    X, y = prepare_data(df)
    
    # Split the data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    print("Training model...")
    model = build_and_train_model(X_train, y_train)
    
    # Evaluate the model
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    print('Model Accuracy:', accuracy_score(y_test, y_pred))
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred))
    
    # Save the model
    print("Saving model...")
    joblib.dump(model, '../models/model.pkl')
    print('Model saved successfully to ../models/model.pkl')

if __name__ == "__main__":
    main() 