import numpy as np
import time
import joblib
from features import features_from_window
from data_gen import generate_example
import argparse

def simulate_window(label, fs=5000, duration=0.5):
    _, v, i = generate_example(label, fs=fs, duration=duration)
    return np.stack([v,i], axis=0)

def load_model(path='models/rf_model.joblib'):
    model, keys = joblib.load(path)
    return model, keys

def infer_stream(mode='simulate', model_path='models/rf_model.joblib', fs=5000, duration=0.5, interval=0.1):
    model, keys = load_model(model_path)
    labels = ['normal','short','overload','open']
    idx = 0

    try:
        while True:
            if mode == 'simulate':
                label = labels[idx % len(labels)]
                idx += 1
                window = simulate_window(label, fs, duration)
            else:
                raise NotImplementedError("ADC mode not implemented")

            vec, keys = features_from_window(window, fs)
            pred = model.predict([vec])[0]
            prob = model.predict_proba([vec])[0] if hasattr(model, "predict_proba") else None

            print(f"Predicted: {pred} | True: {label} | Prob: {prob}")
            time.sleep(interval)

    except KeyboardInterrupt:
        print("Stopped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="simulate")
    parser.add_argument("--model", default="models/rf_model.joblib")
    args = parser.parse_args()
    infer_stream(mode=args.mode, model_path=args.model)
