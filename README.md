# 🫁 PneuXpert AI — Explainable Pneumonia Screening from Chest X-rays

<div align="center">

# AI-Powered Medical Imaging System for Pneumonia Detection

### Fast Screening • Explainable Results • Smarter Healthcare Support

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange)
![Keras](https://img.shields.io/badge/Keras-CNN-red)
![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-ff4b4b)
![GradCAM](https://img.shields.io/badge/Grad--CAM-Explainable%20AI-purple)
![Status](https://img.shields.io/badge/Status-Portfolio%20Project-brightgreen)

</div>

---

## Project Overview

**PneuXpert AI** is an AI-powered medical imaging project designed to detect **Pneumonia from Chest X-ray images** using deep learning.

This project goes beyond a basic image classification model. Instead of only predicting whether an X-ray is **Normal** or **Pneumonia**, it is designed as a complete **Explainable AI Medical Screening System**.

The system can classify chest X-ray images, generate confidence scores, assign risk levels, visualize model attention using Grad-CAM heatmaps, and produce an AI-assisted screening report.

> **Not just a pneumonia classifier — PneuXpert AI is a complete explainable medical imaging workflow for AI-assisted chest X-ray screening.**

---

## What Problem Are We Solving?

Pneumonia is a serious lung infection that can become dangerous if it is not detected early. Chest X-ray imaging is one of the most common tools used by doctors and radiologists to identify signs of pneumonia.

However, real-world healthcare systems may face challenges such as:

- High patient load
- Delay in X-ray review
- Limited access to expert radiologists
- Human fatigue during manual screening
- Need for faster preliminary analysis
- Lack of explainability in AI predictions

This project aims to support faster and more explainable pneumonia screening using deep learning and computer vision.

---

## Why This Project Matters

Most basic machine learning projects only show a simple output:

```text
Prediction: Pneumonia

But a practical healthcare AI system should answer more important questions:

```text
What is the prediction?
How confident is the model?
Which image region influenced the prediction?
What is the risk level?
Can the result be converted into a useful screening report?
```

That is why **PneuXpert AI** is designed as a complete AI screening pipeline instead of a simple CNN demo.

---

## Key Objectives

* Build a deep learning model for pneumonia detection
* Classify chest X-rays into **Normal** and **Pneumonia**
* Use image preprocessing for better model input quality
* Evaluate performance using healthcare-relevant metrics
* Focus on **F1-score** and **recall**, not only accuracy
* Add **Grad-CAM explainability** to visualize model focus areas
* Generate confidence-based risk levels
* Create an AI-assisted medical screening report
* Build a professional GitHub-ready healthcare AI project
* Prepare the project for future deployment as a web app

---

## Main Features

| Feature                    | Description                                                          |
| -------------------------- | -------------------------------------------------------------------- |
| Chest X-ray Classification | Detects whether the uploaded X-ray is Normal or Pneumonia            |
| Deep Learning Model        | Uses CNN-based image classification                                  |
| Transfer Learning Ready    | Can be upgraded with ResNet50, DenseNet121, or EfficientNetB0        |
| Confidence Score           | Shows how confident the model is in its prediction                   |
| Risk Level                 | Converts confidence score into Low, Medium, or High risk             |
| Grad-CAM Explainability    | Highlights suspicious image regions that influenced the model        |
| AI Screening Report        | Generates a structured AI-assisted screening report                  |
| Model Evaluation           | Includes accuracy, precision, recall, F1-score, and confusion matrix |
| Web App Ready              | Can be deployed using Streamlit or FastAPI                           |
| Portfolio Friendly         | Clean structure for GitHub, resume, and internship showcase          |

---

## What Makes This Project Unique?

A normal project works like this:

```text
Upload X-ray
    ↓
Model Prediction
    ↓
Normal / Pneumonia
```

PneuXpert AI works like this:

```text
Upload Chest X-ray
    ↓
Image Preprocessing
    ↓
Deep Learning Prediction
    ↓
Confidence Score
    ↓
Risk Level Calculation
    ↓
Grad-CAM Heatmap
    ↓
AI Screening Report
    ↓
Doctor Review Support
```

This makes the project more realistic, explainable, and impressive for AI/ML portfolios.

---

## Tech Stack

| Category             | Technology                  |
| -------------------- | --------------------------- |
| Programming Language | Python                      |
| Deep Learning        | TensorFlow, Keras           |
| Computer Vision      | OpenCV, PIL                 |
| Data Handling        | NumPy, Pandas               |
| Visualization        | Matplotlib, Seaborn         |
| Evaluation           | Scikit-learn                |
| Explainable AI       | Grad-CAM                    |
| Web App              | Streamlit                   |
| Report Generation    | FPDF / ReportLab            |
| Deployment Ready     | Hugging Face Spaces, Render |
| Version Control      | Git, GitHub                 |

---

## Dataset

The project uses chest X-ray images categorized into two classes:

```text
NORMAL
PNEUMONIA
```

Recommended dataset structure:

```text
dataset/
│
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
├── test/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
└── val/
    ├── NORMAL/
    └── PNEUMONIA/
```

---

## Project Architecture

```text
PneuXpert-AI/
│
├── app/
│   ├── streamlit_app.py
│   └── pages/
│       ├── 1_Predict.py
│       ├── 2_Explainability.py
│       └── 3_Report.py
│
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│   ├── gradcam.py
│   ├── evaluation.py
│   └── report_generator.py
│
├── notebooks/
│   └── pneumonia_detection_experiment.ipynb
│
├── models/
│   └── best_model.h5
│
├── results/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── training_accuracy.png
│   ├── training_loss.png
│   └── gradcam_output.png
│
├── assets/
│   ├── banner.png
│   ├── demo_screenshot.png
│   └── workflow.png
│
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## System Workflow

```text
Chest X-ray Image
        ↓
Image Upload
        ↓
Image Preprocessing
        ↓
Resize + Normalize
        ↓
Deep Learning Model
        ↓
Prediction Result
        ↓
Confidence Score
        ↓
Risk Level Calculation
        ↓
Grad-CAM Heatmap
        ↓
AI Screening Report
        ↓
Final Output
```

---

## Model Pipeline

```text
Input Image
    ↓
Resize Image to 224x224
    ↓
Normalize Pixel Values
    ↓
Pass Image to CNN Model
    ↓
Extract Prediction Probability
    ↓
Classify as Normal or Pneumonia
    ↓
Generate Confidence Score
    ↓
Calculate Risk Level
    ↓
Display Final Result
```

---

## Model Architecture

The base model uses a Convolutional Neural Network for binary image classification.

```text
Input Chest X-ray Image
        ↓
Convolution Layer
        ↓
ReLU Activation
        ↓
Max Pooling
        ↓
Convolution Layer
        ↓
ReLU Activation
        ↓
Max Pooling
        ↓
Flatten Layer
        ↓
Dense Layer
        ↓
Dropout
        ↓
Output Layer
        ↓
Normal / Pneumonia
```

---

## Transfer Learning Upgrade

To make the project stronger and more advanced, the system can be upgraded using transfer learning models.

| Model              | Purpose                                              |
| ------------------ | ---------------------------------------------------- |
| Custom CNN         | Baseline deep learning model                         |
| ResNet50           | Strong computer vision model                         |
| DenseNet121        | Powerful model for medical image classification      |
| EfficientNetB0     | Lightweight and efficient image classification model |
| Vision Transformer | Future advanced upgrade                              |

---

## Evaluation Metrics

Medical AI projects should not depend only on accuracy.

This project focuses on:

| Metric           | Why It Matters                                  |
| ---------------- | ----------------------------------------------- |
| Accuracy         | Measures overall correct predictions            |
| Precision        | Helps reduce false pneumonia predictions        |
| Recall           | Helps reduce missed pneumonia cases             |
| F1-score         | Balances precision and recall                   |
| Confusion Matrix | Shows correct and incorrect predictions clearly |
| ROC Curve        | Shows classification performance                |

---

## Model Performance

| Metric     |                         Score |
| ---------- | ----------------------------: |
| F1-score   |                         94.2% |
| Task Type  |         Binary Classification |
| Classes    |             Normal, Pneumonia |
| Model Type | CNN / Transfer Learning Ready |

---

## Why F1-score Matters

In healthcare AI, accuracy alone is not enough.

A model can have high accuracy but still miss important pneumonia cases. Missing a pneumonia case can be risky because the patient may not receive timely medical attention.

That is why this project highlights **F1-score and recall**, because these metrics are more meaningful for medical screening problems.

---

## Explainable AI with Grad-CAM

Grad-CAM is used to make model predictions more explainable.

Instead of only saying:

```text
Prediction: Pneumonia
```

The system can also show:

```text
Where the model focused inside the X-ray image.
```

Grad-CAM output:

```text
Original X-ray
      ↓
AI Heatmap
      ↓
Highlighted Suspicious Region
```

This improves trust and helps users understand the model decision visually.

---

## Risk Level System

The model prediction confidence is converted into a simple risk level.

| Confidence Score | Risk Level  |
| ---------------: | ----------- |
|         0% - 50% | Low Risk    |
|        51% - 80% | Medium Risk |
|       81% - 100% | High Risk   |

Example:

```text
Prediction: Pneumonia
Confidence: 94.2%
Risk Level: High
Recommendation: Medical review required
```

---

## AI Screening Report

The system can generate a structured AI-assisted screening report.

Example report:

```text
AI Chest X-ray Screening Report

Prediction: Pneumonia
Confidence Score: 94.2%
Risk Level: High

AI Observation:
The model detected abnormal opacity-like visual patterns in the lung region.

Recommendation:
This result should be reviewed by a certified medical professional.

Disclaimer:
This report is generated by an AI model for educational and research purposes only.
It is not a replacement for professional medical diagnosis.
```

---

## Web App Flow

The project can be deployed as a simple web application.

```text
Home Page
    ↓
Upload Chest X-ray
    ↓
Preview Image
    ↓
Run Prediction
    ↓
Show Result
    ↓
Show Confidence Score
    ↓
Show Risk Level
    ↓
Show Grad-CAM Heatmap
    ↓
Download Report
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/2KRISHNAYADAV/Medical-Imaging-Pneumonia-Detection.git
cd Medical-Imaging-Pneumonia-Detection
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

### 3. Activate Virtual Environment

For Windows:

```bash
venv\Scripts\activate
```

For macOS/Linux:

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## How to Train the Model

```bash
python src/train.py
```

---

## How to Run Prediction

```bash
python src/predict.py
```

---

## How to Run Streamlit App

```bash
streamlit run app/streamlit_app.py
```

---

## Expected Output

```text
Uploaded Image: chest_xray_sample.jpeg

Prediction: Pneumonia
Confidence Score: 94.2%
Risk Level: High

AI Recommendation:
Please consult a certified medical professional for final diagnosis.
```

---

## Results Visualization

The project includes visual outputs such as:

```text
Training Accuracy Graph
Training Loss Graph
Confusion Matrix
ROC Curve
Grad-CAM Heatmap
Prediction Report
```

Recommended output folder:

```text
results/
│
├── confusion_matrix.png
├── roc_curve.png
├── training_accuracy.png
├── training_loss.png
└── gradcam_output.png
```

---

## Future Improvements

* Add DenseNet121 transfer learning model
* Add EfficientNetB0 model comparison
* Add Vision Transformer-based classification
* Add Grad-CAM++ for better explainability
* Add PDF report download
* Add FastAPI backend
* Add React dashboard
* Add Docker support
* Add MLflow experiment tracking
* Add DVC for dataset version control
* Deploy on Hugging Face Spaces
* Add model card for responsible AI documentation
* Add multi-disease chest X-ray classification

---

## Learning Outcomes

Through this project, I learned:

* How medical image classification works
* How CNNs process X-ray images
* How to preprocess image datasets
* How to train and evaluate deep learning models
* Why recall and F1-score matter in healthcare AI
* How explainable AI improves trust
* How Grad-CAM highlights important image regions
* How to design a real-world AI screening workflow
* How to structure a professional GitHub ML project

---

## Real-World Use Case

This project can be used as an educational prototype for:

* AI-assisted medical screening
* Pneumonia detection research
* Healthcare AI learning
* Medical image classification practice
* Deep learning portfolio project
* Explainable AI demonstration

---

## Important Disclaimer

This project is created only for **educational and research purposes**.

It is not intended for real clinical diagnosis, treatment, or medical decision-making.

The prediction generated by this model should always be reviewed by a qualified medical professional.

AI can support healthcare experts, but it should not replace doctors or radiologists.

---

## Author

**Krishna Yadav**
B.Tech Computer Science Engineering
AI/ML & Data Science Enthusiast

GitHub: [2KRISHNAYADAV](https://github.com/2KRISHNAYADAV)

---

## Repository Description

```text
Explainable AI system for pneumonia screening from chest X-ray images using CNN, transfer learning, Grad-CAM visualization, confidence scoring, risk triage, and AI screening report generation.
```

---

## Suggested GitHub Topics

```text
deep-learning
medical-imaging
pneumonia-detection
chest-xray
cnn
tensorflow
keras
computer-vision
grad-cam
explainable-ai
healthcare-ai
streamlit
machine-learning
artificial-intelligence
```

---

## Final Thought

> AI in healthcare is not about replacing doctors.
> It is about building intelligent systems that support faster screening, better decisions, and more accessible medical care.

**PneuXpert AI is a step toward explainable and responsible healthcare intelligence.**

```
```
