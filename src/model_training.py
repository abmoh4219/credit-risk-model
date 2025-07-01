import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os

def create_model_pipeline(categorical_cols):
    """Create a pipeline for preprocessing and modeling."""
    numerical_cols = ['Amount', 'Value', 'total_amount', 'avg_amount', 'trans_count', 'std_amount']

    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ])

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42))
    ])

    return pipeline, numerical_cols, categorical_cols

def train_model(input_path, model_path):
    """Train and evaluate a Logistic Regression model."""
    df = pd.read_csv(input_path)
    
    # Check distribution
    print("is_high_risk distribution:", df['is_high_risk'].value_counts(normalize=True))
    
    # Dynamically get categorical columns (one-hot encoded)
    categorical_cols = [col for col in df.columns if col.startswith('ProductCategory_') or col.startswith('ChannelId_')]
    numerical_cols = ['Amount', 'Value', 'total_amount', 'avg_amount', 'trans_count', 'std_amount']
    
    # Prepare features and target
    X = df[numerical_cols + categorical_cols]
    y = df['is_high_risk']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    pipeline, _, _ = create_model_pipeline(categorical_cols)
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipeline.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, y_pred))
    
    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)
    return pipeline

if __name__ == "__main__":
    train_model('data/processed/data_with_proxy.csv', 'models/logistic_model.pkl')
