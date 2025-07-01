from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)
model = joblib.load('models/logistic_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    """Predict credit risk for a single customer."""
    data = request.get_json()
    df = pd.DataFrame([data])
    
    # Required columns (numerical + all one-hot encoded columns)
    required_cols = [
        'Amount', 'Value', 'total_amount', 'avg_amount', 'trans_count', 'std_amount',
        'ProductCategory_airtime', 'ProductCategory_data_bundles',
        'ProductCategory_financial_services', 'ProductCategory_movies',
        'ProductCategory_other', 'ProductCategory_ticket',
        'ProductCategory_transport', 'ProductCategory_tv',
        'ProductCategory_utility_bill', 'ChannelId_ChannelId_1',
        'ChannelId_ChannelId_2', 'ChannelId_ChannelId_3', 'ChannelId_ChannelId_5'
    ]
    for col in required_cols:
        if col not in df:
            df[col] = 0  # Fill missing one-hot columns with 0
    
    # Predict
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    
    return jsonify({
        'is_high_risk': int(prediction),
        'probability': float(probability)
    })

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
