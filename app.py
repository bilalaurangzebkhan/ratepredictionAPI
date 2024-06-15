from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# Load the entire model
model = load_model("prediction_model.h5")

# Load the tokenizer and scaler
with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

df = pd.read_csv("wheat_dataset.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date', ascending=False)

# Create lag features function
def create_lag_features(df, lag=2):
    lag_df = pd.DataFrame()
    for i in range(1, lag + 1):
        df[f'Time lag {i}'] = df.groupby('District/City')['Minimum Rate'].shift(i)
    lag_df = pd.concat([lag_df, df.dropna()], ignore_index=True)
    return lag_df

# Preprocess the city name and create lag features for prediction
def preprocess_city(city_name):
    city_name = city_name.strip().lower()

    city_sequence = tokenizer.texts_to_sequences([city_name])
    city_sequence = pad_sequences(city_sequence, maxlen=1)

    city_data = df[df['District/City'].str.lower() == city_name]
    if city_data.empty or len(city_data) < 2:
        return None, None

    lag_df = create_lag_features(city_data, lag=2)

    time_lag_1 = lag_df.iloc[0]['Time lag 1']
    time_lag_2 = lag_df.iloc[0]['Time lag 2']

    lag_features = np.array([[time_lag_1, time_lag_2]])
    lag_features = scaler.transform(lag_features)

    return city_sequence, lag_features

# Prediction route
@app.route('/predict', methods=['GET','POST'])
def predict_price():
    data = request.get_json(force=True)
    city_name = data['city_name'].strip().lower()

    def get_previous_rates(city_name):
        city_data = df[df['District/City'].str.lower() == city_name].head(7)
        previous_rates = []
        for index, row in city_data.iterrows():
            previous_rates.append({
                "City": city_name,
                "Date": row['Date'].strftime('%a, %d %b %Y'),
                "Minimum Rate": row['Minimum Rate']
            })
        return previous_rates

    city_sequence, lag_features = preprocess_city(city_name)
    if city_sequence is None or lag_features is None:
        return jsonify({"error": "Not enough data for the city to predict price."}), 400

    predicted_price = model.predict([lag_features, city_sequence])

    # Format the predicted price to two decimal places and handle rounding
    predicted_value = float(predicted_price[0][0])
    if predicted_value % 1 >= 0.50:
        formatted_price = round(predicted_value)
    elif predicted_value % 1 <= 0.50:
        formatted_price = round(predicted_value)
    else:
        formatted_price = round(predicted_value, 2)

    # Get previous 7 days' rates
    previous_rates = get_previous_rates(city_name)

    return jsonify({"predicted_price": formatted_price, "previous_rates": previous_rates})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)