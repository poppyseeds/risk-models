import tensorflow as tf
import numpy as np

# Create a compatible network detection model (41 inputs)
network_model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(41,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
network_model.compile(optimizer='adam', loss='binary_crossentropy')
network_model.save('models/ram_model.h5')
print("✅ Created ram_model.h5")

# Create a compatible LSTM model (3 input features, window of 10)
lstm_model = tf.keras.Sequential([
    tf.keras.layers.LSTM(32, activation='relu', input_shape=(10, 3), return_sequences=True),
    tf.keras.layers.LSTM(16, activation='relu'),
    tf.keras.layers.Dense(3)  # Reconstruct 3 features
])
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.save('models/lstm_model.h5')
print("✅ Created lstm_model.h5")
