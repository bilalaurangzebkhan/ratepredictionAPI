import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Embedding, Concatenate, Reshape, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError, RootMeanSquaredError, MeanAbsolutePercentageError
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Read the dataset
df = pd.read_csv("wheat_dataset.csv")

# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

def create_lag_features(df, lag=2):
    lag_df = pd.DataFrame()
    for i in range(1, lag + 1):
        # Shift the 'Minimum Rate' values by one day within each city group
        df[f'Time lag {i}'] = df.groupby('District/City')['Minimum Rate'].shift(i)
    lag_df = pd.concat([lag_df, df.dropna()], ignore_index=True)
    return lag_df

# Create lag features
lag_df = create_lag_features(df, lag=2)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['District/City'])
city_sequences = tokenizer.texts_to_sequences(df['District/City'])
city_sequences = pad_sequences(city_sequences, maxlen=1)

# Normalize the features
scaler = MinMaxScaler()
lag_df[[f'Time lag {i}' for i in range(1, 3)]] = scaler.fit_transform(lag_df[[f'Time lag {i}' for i in range(1, 3)]])

# Data Splitting
X = lag_df[['Time lag 1', 'Time lag 2']].values
y = lag_df['Minimum Rate'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Model Building
city_input = Input(shape=(1,))
city_embedding = Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=10)(city_input)
city_embedding = Flatten()(city_embedding)

lag_input = Input(shape=(2,))
concatenated = Concatenate()([lag_input, city_embedding])
reshaped = Reshape((1, -1))(concatenated)
lstm = LSTM(units=64, return_sequences=True)(reshaped)
lstm = LSTM(units=64, return_sequences=False)(lstm)
dense = Dense(units=32, activation='relu')(lstm)
dropout = Dropout(0.2)(dense)
output = Dense(units=1)(dropout)

model = Model(inputs=[lag_input, city_input], outputs=output)

# Compiling the model with metrics
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error',
              metrics=[MeanSquaredError(), MeanAbsoluteError(), RootMeanSquaredError(), MeanAbsolutePercentageError()])

# Model Training
history = model.fit([X_train, city_sequences[:len(X_train)]], y_train, batch_size=32, epochs=50, verbose=1, validation_split=0.1)  # Use verbose=1 for training progress

history_df = pd.DataFrame(history.history)
history_df.to_csv("training_history.csv", index=False)

model.save("prediction_model.h5")
