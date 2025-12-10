# AI-Powered Fault Detection in Electrical Networks  
### Real-Time Signal Processing + ML Classification + Web Dashboard

This project implements an end-to-end **AI-based electrical fault detection system** using:

- Synthetic waveform generation (Normal / Short / Overload / Open)
- Signal processing feature extraction (FFT, THD, spectral metrics)
- ML models (RandomForest + SVM)
- Real-time streaming inference
- Web dashboard with waveform visualization

Perfect for **portfolio**, **demonstrations**, **research**, or **interview projects**.

---

## 🚀 Features

### ✔ Synthetic Data Generator
Generates voltage/current waveforms for:
- Normal operation  
- Short-circuit faults  
- Overload  
- Open-circuit

### ✔ Feature Engineering
Extracts:
- RMS, Peak, Mean
- THD (Total Harmonic Distortion)
- Spectral centroid
- FFT band energies
- Kurtosis, Zero Crossing Rate

### ✔ Machine Learning Pipeline
Trains:
- RandomForest  
- SVM  
Saves models using joblib.

### ✔ Real-Time Inference
Supports:
- Simulated signals  
- ADC hardware (ADS1115 / ESP32 stream)

### ✔ Web Dashboard
Flask + Chart.js frontend:
- Shows waveform
- Displays predicted fault class + probabilities

---

## 📦 Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
