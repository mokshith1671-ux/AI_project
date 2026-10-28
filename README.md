# AI-Based Indoor Navigation Using Wi-Fi Fingerprinting & Machine Learning

An AI-powered indoor localization and navigation system comparing Classical Machine Learning (KNN, Random Forest, XGBoost) and Deep Learning (Multi-Layer Perceptron) architectures on Wi-Fi Received Signal Strength Indicator (RSSI) fingerprints.

## Project Structure
```
AI_project/
├── path1_pipeline.py                  # Custom building dataset pipeline (60 fingerprints)
├── WiFi_Indoor_Navigation_All_Sessions.xlsx  # Custom building raw Wi-Fi dataset
├── path2_benchmark/                   # Open-source benchmark pipeline
│   ├── download_dataset.py            # Automated downloader for UJIIndoorLoc dataset
│   └── path2_pipeline.py              # ML/DL benchmark & multi-floor navigation engine
├── requirements.txt                   # Project dependencies
├── .gitignore                         # Git ignore rules
└── README.md                          # Project documentation
```

## Features
- **Wi-Fi Fingerprint Preprocessing:** Automated cleaning, missing signal sentinel handling (-105 dBm), and feature scaling (MinMaxScaler).
- **ML/DL Model Comparison:** Evaluation across KNN, Random Forest, XGBoost, and Deep Learning Neural Networks (MLP).
- **Pathfinding Engine:** Graph-based multi-floor route generation using Dijkstra's algorithm.
- **Performance Metrics:** Accuracy, Macro F1-score, and confusion matrix visual reports.

## Benchmark Results (Path 2 - UJIIndoorLoc)
- **Top Classifier:** K-Nearest Neighbors & Random Forest (**~96.83% Validation Accuracy**)
- **XGBoost:** 95.90% Accuracy
- **Neural Network (MLP):** ~95%+ Accuracy with MinMaxScaler normalization

## Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/mokshith1671-ux/AI_project
cd AI_project
```

### 2. Create & Activate Virtual Environment
- **Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```
- **macOS / Linux:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Running the Pipelines

> **Note:** Make sure your virtual environment is activated before running any scripts (`.venv` should be visible in your terminal prompt).

#### Path 1: Custom Building Prototype (Run from Project Root `AI_project/`)
# Ensure you are in the AI_project root directory

python path1_pipeline.py

### Path 2: UJIIndoorLoc Benchmark Evaluation
# Navigate to the benchmark directory

cd path2_benchmark

# Download and extract the dataset (Run once)
python download_dataset.py

# Execute ML/DL benchmarking and navigation engine
python path2_pipeline.py

