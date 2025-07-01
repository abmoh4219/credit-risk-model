import pandas as pd

def inspect_rfm(input_path):
    """Inspect RFM metrics to diagnose proxy variable issues."""
    df = pd.read_csv(input_path)
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    max_date = df['TransactionStartTime'].max()
    
    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (max_date - x.max()).days,  # Recency
        'TransactionId': 'count',  # Frequency
        'Amount': 'sum'  # Monetary
    }).reset_index()
    
    rfm.columns = ['CustomerId', 'recency', 'frequency', 'monetary']
    
    print("RFM Summary Statistics:")
    print(rfm.describe())
    print("\nQuantiles (0.20, 0.30, 0.50, 0.70, 0.80):")
    print(rfm[['recency', 'frequency', 'monetary']].quantile([0.20, 0.30, 0.50, 0.70, 0.80]))

if __name__ == "__main__":
    inspect_rfm('data/raw/data.csv')
