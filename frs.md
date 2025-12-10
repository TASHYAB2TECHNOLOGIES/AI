# Functional Requirements Specification (FRS)

## 1. Data Generation Requirements
- **FRS-DG-1:** Generate voltage/current waveforms for four classes  
- **FRS-DG-2:** Add noise & harmonic distortion  
- **FRS-DG-3:** Save dataset in `.npz` format  

## 2. Feature Extraction Requirements
- **FRS-FE-1:** Compute time-domain features: RMS, Peak, Mean, Kurtosis, ZCR  
- **FRS-FE-2:** Compute frequency-domain features: FFT, THD, Spectral centroid, Band energies  
- **FRS-FE-3:** Return features as numeric vector  

## 3. ML Model Requirements
- **FRS-ML-1:** Split dataset into train/test sets  
- **FRS-ML-2:** Train RandomForest & SVM classifiers  
- **FRS-ML-3:** Generate classification report & confusion matrix  
- **FRS-ML-4:** Save trained models using joblib  

## 4. Real-Time Inference Requirements
- **FRS-RT-1:** Support simulated streaming windows  
- **FRS-RT-2:** Support ADC real input (optional)  
- **FRS-RT-3:** Compute features for each window  
- **FRS-RT-4:** Produce fault predictions & probability scores  

## 5. Web Dashboard Requirements
- **FRS-UI-1:** Display waveform data  
- **FRS-UI-2:** Show predicted fault label  
- **FRS-UI-3:** Refresh at least once per second  
- **FRS-UI-4:** Server endpoints: `/predict_simulate`, `/predict_upload`, `/stream`  

## 6. System Configuration Requirements
- **FRS-CFG-1:** Use YAML/JSON config (future)  
- **FRS-CFG-2:** Allow user to set sampling rate & window duration  
