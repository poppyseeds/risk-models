from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# ===== LOAD MODELS =====
network_model = tf.keras.models.load_model(
    "models/ram_model.h5",
    compile=False
)
network_scaler = joblib.load("models/scaler.pkl")
columns = joblib.load("models/columns.pkl")

lstm_model = tf.keras.models.load_model(
    "models/lstm_model.h5",
    compile=False
)
lstm_scaler = joblib.load("models/lstm_scaler.pkl")
threshold = np.load("models/threshold.npy")

# ===== HELPERS =====
def prepare_network_input(data_dict):
    import pandas as pd
    
    df = pd.DataFrame([data_dict])

    # add missing columns with 0
    for col in columns:
        if col not in df.columns:
            df[col] = 0

    # keep only required columns in correct order
    df = df[columns]

    return df

def detect_network(data_dict):
    df = prepare_network_input(data_dict)
    scaled = network_scaler.transform(df)
    pred = network_model.predict(scaled)
    return "🚨 THREAT" if pred[0] == -1 else "✅ NORMAL"

def create_sequence(data, window=10):
    return np.array([data[-window:]])

def detect_process(data):
    data = np.array(data)
    scaled_data = lstm_scaler.transform(data)
    seq = create_sequence(scaled_data)

    recon = lstm_model.predict(seq, verbose=0)
    loss = np.mean((seq - recon)**2)

    return "🚨 ANOMALY" if loss > threshold else "✅ NORMAL"

# ===== ROUTES =====
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    req = request.json

    network_data = req["network"]
    process_data = req["process"]

    net_result = detect_network(network_data)
    proc_result = detect_process(process_data)

    return jsonify({
        "network": net_result,
        "process": proc_result
    })

# ===== RUN =====
if __name__ == "__main__":
    app.run(debug=True)