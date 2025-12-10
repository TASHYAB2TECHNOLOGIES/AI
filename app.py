from flask import Flask, jsonify, request, send_from_directory
import joblib
import numpy as np
import time
from features import features_from_window
from data_gen import generate_example

app = Flask(__name__, static_folder='../web', static_url_path='/')

MODEL_PATH = 'models/rf_model.joblib'
model, keys = joblib.load(MODEL_PATH)

last = {"ts": None, "wave_v": None, "wave_i": None, "pred": None, "prob": None}

@app.route("/")
def index():
    return send_from_directory('../web', 'index.html')

@app.route("/predict_simulate")
def predict_simulate():
    label = request.args.get("label", "normal")
    t, v, i = generate_example(label, fs=5000, duration=0.5)
    window = np.stack([v,i], axis=0)
    vec, _ = features_from_window(window)
    pred = model.predict([vec])[0]
    prob = model.predict_proba([vec])[0].tolist() if hasattr(model, "predict_proba") else None

    global last
    last = {"ts": time.time(), "wave_v": v.tolist(), "wave_i": i.tolist(), "pred": pred, "prob": prob}
    return jsonify(last)

@app.route("/stream")
def stream():
    return jsonify(last)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
