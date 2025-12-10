# Software Requirements Specification (SRS)
## 1. Purpose
Defines functional, non-functional, and system requirements for the AI-Powered Fault Detection in Electrical Networks system.

- Generate synthetic electrical waveforms  
- Extract signal-processing features  
- Train machine learning models  
- Classify electrical faults  
- Perform real-time inference  
- Visualize results on a web dashboard  

## 2. Scope
Provides a complete pipeline for fault detection in low-voltage or simulated electrical networks. Fault types include:

- Normal  
- Short Circuit  
- Overload  
- Open Circuit  

## 3. Definitions
| Term | Definition |
|------|------------|
| ADC | Analog to Digital Converter |
| FFT | Fast Fourier Transform |
| THD | Total Harmonic Distortion |
| CT | Current Transformer |
| Fault | Abnormal electrical condition (short, overload, open) |

## 4. References
- IEEE 830 SRS standard  
- scikit-learn documentation  
- Flask documentation  

## 5. Overall Description

### 5.1 Product Perspective
Standalone software for laptops, Raspberry Pi, or edge devices.  
Interacts with:  
- Simulated waveform generator  
- Real ADC hardware (optional)  
- Web dashboard  

### 5.2 Product Features
- Signal generation & labeling  
- Feature extraction (RMS, THD, FFT spectral energy, etc.)  
- ML model training (RandomForest, SVM)  
- Real-time inference  
- REST API endpoints  
- Live waveform visualization  

### 5.3 User Classes
| User | Description |
|------|-------------|
| Researcher | Experiments with fault detection |
| ML Engineer | Trains models & improves accuracy |
| Technician | Monitors dashboard |
| Recruiter | Evaluates candidate project |

## 6. Specific Requirements

### 6.1 Functional Requirements
See FRS.md

### 6.2 Non-Functional Requirements
- Performance: Process each inference window < 100 ms  
- Reliability: Model accuracy > 90% on synthetic dataset  
- Usability: Dashboard requires no technical expertise  
- Security: Sanitize all inputs  
- Portability: Runs on Windows, Linux, macOS, Raspberry Pi  

## 7. External Interface Requirements
- **User Interface:** Web UI (HTML + Chart.js) showing waveform plot and fault label.  
- **Hardware Interface:** Optional ADC: ADS1115, ESP32  
- **Software Interface:** Python REST API endpoints: `/predict_simulate`, `/predict_upload`, `/stream`  

## 8. System Requirements
- Python 3.9+  
- 1 GB RAM minimum  
- Optional hardware drivers for ADC  
