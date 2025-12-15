# EV Battery Fault Detection using Neural Networks
**ECE 5831 – Pattern Recognition & Neural Networks (Final Project)**  
**Year:** 2025  

---

## Project Overview
This project focuses on fault detection in lithium-ion batteries using neural networks and real experimental data. The objective is to identify abnormal battery behavior (capacity fade / fault conditions) from discharge measurements using deep learning models applied to time-series signals.

The project uses the NASA Li-ion Battery Aging Dataset, performs preprocessing and sliding-window feature extraction, and evaluates multiple neural network architectures. Emphasis is placed on reproducibility and proper test evaluation using batteries that contain fault samples.

---

## Key Contributions
- Real-world battery aging dataset (NASA)
- Sliding-window time-series modeling
- CNN, LSTM, and TCN architecture comparison
- Threshold tuning to improve fault detection
- Evaluation using Precision–Recall and ROC curves
- Fully reproducible pipeline (CLI + notebooks)

---

## Repository Structure

```
ece5831-2025-final-project/
│
├── Data/
│ └── battery_data/ # NASA .mat battery files
│
├── src/ # Core pipeline code
│ ├── preprocess.py # Load .mat files and generate labels
│ ├── windows.py # Sliding-window generation
│ ├── train.py # Model training
│ ├── evaluate.py # Test evaluation + plots
│ ├── thresholds.py # Threshold tuning
│ └── config.py # Central configuration
│
├── report/
│ └── final_report.pdf # IEEE-format final report
│
├── slides/
│ └── final_presentation.pptx # Presentation slides
│
├── final-project.ipynb
├── final-project_ThreeModelTraining.ipynb
├── README.md
└── requirements.txt
```

---

## Dataset
- **Source:** NASA Battery Dataset  
  https://www.kaggle.com/datasets/nuheel/battery-health-nasa-dataset

- **Batteries Used:** B0005, B0006, B0007, B0018  
- **Primary Test Battery:** B0018 (contains fault samples)  
- **Note:** Battery B0007 contains no fault samples and is excluded from primary test evaluation.

---

## How to Run the Project (Reproducibility)

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/ece5831-2025-final-project
cd ece5831-2025-final-project

### 2. Create a clean Python environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

### 3. Run the pipeline from the command line
python -m src.preprocess
python -m src.windows
python -m src.train --model cnn1d --epochs 10 --out runs/example_run
python -m src.evaluate --ckpt runs/example_run/best.pt --out runs/example_run

### 4. Run the notebooks
Open and execute:

final-project.ipynb
final-project_ThreeModelTraining.ipynb

All cells are pre-executed and saved.

### Models Implemented
CNN1D: Baseline temporal convolution model
LSTM: Recurrent neural network for sequence modeling
TCN: Dilated temporal convolutional network
# Performance metrics include:
Average Precision (AP)
Precision, Recall, and F1 score
Precision–Recall and ROC curves

### Final Project Deliverables
# Presentation Slides

https://github.com/
<your-username>/ece5831-2025-final-project/blob/main/slides/final_presentation.pptx

# Presentation Video (15 minutes)

https://youtu.be/XXXXXXXXXXX

# Final Report (IEEE Format)

https://github.com/
<your-username>/ece5831-2025-final-project/blob/main/report/final_report.pdf

# 📁 Dataset

Original source: https://www.kaggle.com/datasets/nuheel/battery-health-nasa-dataset

Local copy: Data/battery_data/

# Demo Video

https://youtu.be/YYYYYYYYYYY

### Reproducibility Notes

Tested on a clean Python virtual environment using venv

Requires only pip install -r requirements.txt

No proprietary tools required

# Author
Sobhi Zeidan
ECE 5831 – University of Michigan