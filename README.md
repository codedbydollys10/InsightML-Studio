# 🚀 InsightML Studio



[![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red?style=for-the-badge&logo=streamlit)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikitlearn)](https://scikit-learn.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge&logo=opencv)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Vision-blueviolet?style=for-the-badge)](https://mediapipe.dev/)
[![License](https://img.shields.io/badge/License-MIT-success?style=for-the-badge)](LICENSE)

---

## 🌟 Overview

InsightML Studio is an all-in-one Machine Learning and Computer Vision platform built with Python, Streamlit, Scikit-Learn, OpenCV, and MediaPipe.

It enables users to perform complete end-to-end ML workflows—from uploading raw datasets to generating predictions, visualizations, explainability reports, business insights, and computer vision analysis—within a modern interactive web interface.

Whether you're a student, researcher, data scientist, or developer, InsightML Studio provides an intuitive environment to explore data, build models, analyze results, and generate actionable insights.

---

## ✨ Features

### 📂 Dataset Explorer
- Upload CSV datasets
- Automatic dataset profiling
- Data type detection
- Missing value analysis
- Duplicate detection
- Dataset statistics
- Memory usage
- Column summaries

### 📊 Automated EDA
Automatically generates:
- Distribution plots
- Correlation heatmaps
- Missing value visualization
- Outlier detection
- Class distribution
- Feature statistics
- Pairwise relationships
- Numerical summaries

### ⚙ Data Preprocessing
Supports:
- Missing value handling
- Label encoding
- One-hot encoding
- Feature scaling
- Feature selection
- Duplicate removal
- Data cleaning

### 🧠 Feature Engineering
Automatically creates useful features such as:
- Banking domain features
- Ratio features
- Log transformations
- Balance change metrics
- Credit/Debit ratios
- Variance-based feature selection

### 🤖 Machine Learning
Supports:
- Classification
- Regression

Automatically trains multiple models:
- Logistic Regression
- Linear Regression
- Decision Tree
- Random Forest
- Extra Trees
- Gradient Boosting
- AdaBoost
- KNN
- Support Vector Machine
- Naive Bayes
- XGBoost
- LightGBM
- CatBoost
- MLP Neural Network

### 🏆 Model Comparison
Automatically compares models using cross-validation and displays:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC
- MAE
- RMSE
- R² Score
- Training time
- Cross validation score

### 📈 Hyperparameter Optimization
Supports:
- Cross validation
- Grid search
- Randomized search
- Best model selection

### 🎯 Prediction
Make predictions using trained models:
- Single prediction
- Batch prediction
- CSV upload prediction

### 📦 Batch Prediction
Upload a CSV with unseen data and generate:
- Prediction CSV
- Downloadable results
- Prediction summary

### 🧠 Explainable AI (XAI)
Understand model decisions using:
- SHAP values
- Feature importance
- Global explainability
- Local explainability
- Decision interpretation

### 📈 Business Insights
Automatically generates business-friendly insights:
- Customer segmentation
- Risk analysis
- Revenue opportunities
- Important feature highlights
- Strategic recommendations

### 📥 Downloads
Generate downloadable:
- Processed dataset
- Predictions
- Reports
- Model leaderboard
- Feature importance
- Evaluation results

### 👁 Vision AI
Includes a computer vision module for image analysis.

#### 👤 Human Analysis
Using MediaPipe, detects:
- Face
- Eyes
- Nose
- Lips
- Ears
- Face mesh
- Facial landmarks

#### 🚗 Object Detection
Detects objects like:
- Cars
- Bikes
- Trucks
- Buses
- People
- Animals

#### 🌳 Scene Analysis
Analyzes:
- Nature
- Forest
- Mountain
- Beach
- City
- Indoor
- Sky
- Sunset

#### 🖼 Image Analysis
Automatically computes:
- Resolution
- Brightness
- Contrast
- Sharpness
- Blur score

#### 📄 Vision Report
Generate complete image analysis reports including:
- Human detection
- Object detection
- Scene analysis
- Image quality
- Confidence scores

---

## 📁 Project Structure

```
InsightML-Studio/
├── app.py
├── config.py
├── requirements.txt
├── README.md
├── LICENSE
├── artifacts/
├── data/
├── models/
│   └── vision/
├── notebooks/
├── pages/
└── utils/
```

---

## 🛠 Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- LightGBM
- CatBoost
- OpenCV
- MediaPipe
- Plotly
- SHAP
- Joblib

---

## 🚀 Installation

```bash
git clone https://github.com/codedbydollys10/InsightML-Studio.git
cd InsightML-Studio
python -m venv venv
```

### Activate virtual environment

Windows:
```bash
venv\Scripts\activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the app:
```bash
streamlit run app.py
```

---

## 🔮 Future Enhancements

- AutoML
- Time series forecasting
- NLP module
- OCR support
- Video analysis
- Audio analysis
- Live webcam detection
- Model monitoring
- Cloud deployment
- REST API support

---

## 📜 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

Dolly Sharma  
GitHub: [github.com/codedbydollys10](https://github.com/codedbydollys10)

---

⭐ If you found this project helpful, consider giving it a star on GitHub!
