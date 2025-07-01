import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

def extract_temporal_features(df):
    """Extract hour, day, month, and year from TransactionStartTime."""
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    df['trans_hour'] = df['TransactionStartTime'].dt.hour
    df['trans_day'] = df['TransactionStartTime'].dt.day
    df['trans_month'] = df['TransactionStartTime'].dt.month
    df['trans_year'] = df['TransactionStartTime'].dt.year
    return df

def create_aggregate_features(df):
    """Create aggregate features per CustomerId."""
    agg_features = df.groupby('CustomerId').agg({
        'Amount': ['sum', 'mean', 'count', 'std'],
        'TransactionId': 'nunique'
    }).reset_index()
    agg_features.columns = ['CustomerId', 'total_amount', 'avg_amount', 'trans_count', 'std_amount', 'unique_trans']
    return agg_features

def create_feature_pipeline():
    """Create a pipeline for preprocessing numerical and categorical features."""
    numerical_cols = ['Amount', 'Value', 'total_amount', 'avg_amount', 'trans_count', 'std_amount']
    categorical_cols = ['ProductCategory', 'ChannelId']

    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ])

    return preprocessor, numerical_cols, categorical_cols

def process_data(input_path, output_path):
    """Process raw data and save model-ready data."""
    df = pd.read_csv(input_path)
    df = extract_temporal_features(df)
    agg_df = create_aggregate_features(df)
    df = df.merge(agg_df, on='CustomerId', how='left')
    pipeline, numerical_cols, categorical_cols = create_feature_pipeline()
    transformed_data = pipeline.fit_transform(df)
    
    # Get column names for transformed data
    cat_cols = pipeline.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_cols)
    all_cols = numerical_cols + list(cat_cols)
    
    # Convert to DataFrame and retain CustomerId
    transformed_df = pd.DataFrame(transformed_data, columns=all_cols)
    transformed_df['CustomerId'] = df['CustomerId'].reset_index(drop=True)
    transformed_df.to_csv(output_path, index=False)
    return transformed_df

if __name__ == "__main__":
    process_data('data/raw/data.csv', 'data/processed/processed_data.csv')
