import pandas as pd
import numpy as np

def calculate_rfm(df):
    """Calculate Recency, Frequency, and Monetary metrics per CustomerId."""
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    max_date = df['TransactionStartTime'].max()
    
    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (max_date - x.max()).days,  # Recency
        'TransactionId': 'count',  # Frequency
        'Amount': 'sum'  # Monetary
    }).reset_index()
    
    rfm.columns = ['CustomerId', 'recency', 'frequency', 'monetary']
    return rfm

def create_proxy_variable(raw_path, processed_path, output_path):
    """Create is_high_risk proxy variable and merge with processed data."""
    # Load raw data for RFM
    raw_df = pd.read_csv(raw_path)
    rfm = calculate_rfm(raw_df)
    
    # Thresholds for ~15-25% high-risk (0.85 quantiles)
    recency_threshold = rfm['recency'].quantile(0.85)  # Top 15% most recent (~63 days)
    monetary_threshold = rfm['monetary'].quantile(0.85)  # Top 15% monetary (~150,000)
    
    # High-risk: recent transactions OR high monetary
    rfm['is_high_risk'] = ((rfm['recency'] <= recency_threshold) | 
                          (rfm['monetary'] >= monetary_threshold)).astype(int)
    
    # Print distribution before merge
    print("High-risk distribution (before merge):", rfm['is_high_risk'].value_counts(normalize=True))
    
    # Load processed data and merge
    processed_df = pd.read_csv(processed_path)
    processed_df = processed_df.merge(rfm[['CustomerId', 'is_high_risk']], on='CustomerId', how='left')
    processed_df['is_high_risk'] = processed_df['is_high_risk'].fillna(0).astype(int)
    
    # Print distribution after merge
    print("High-risk distribution (after merge):", processed_df['is_high_risk'].value_counts(normalize=True))
    
    # Save to output
    processed_df.to_csv(output_path, index=False)
    return processed_df

if __name__ == "__main__":
    create_proxy_variable('data/raw/data.csv', 'data/processed/processed_data.csv', 'data/processed/data_with_proxy.csv')
