# Branch ML Project: Loan Default Prediction

Welcome to the **Branch ML Project**, focused on predicting loan defaults using machine learning techniques. This project leverages an SVM classifier, provides a FastAPI-based prediction service, and incorporates geospatial data analysis to deliver comprehensive insights.

---

## 📦 Project Components

- **SVM Classifier for Loan Default Prediction:** Utilizes a Support Vector Machine (SVM) to accurately predict the likelihood of loan defaults based on various features.
- **FastAPI Prediction Service:** A robust API built with FastAPI to serve real-time predictions using the trained SVM model.
- **Geospatial Data Analysis:** Analyzes and visualizes geospatial data to uncover patterns and trends related to loan defaults.

---

## 🗂️ Key Artifacts

- **`svm_pipeline.joblib`:** The serialized and trained SVM model ready for deployment.
- **`app.py`:** The FastAPI application script that serves the prediction API.
- **`geoplot.py`:** Script for geospatial visualization of data and model insights.
- **`svm.py`:** Model training script that includes Principal Component Analysis (PCA) for dimensionality reduction.

---

## 🛠️ Data Processing

- **Source:** Data is sourced from a PostgreSQL database, encompassing a rich set of features.
- **Features:** Includes GPS data, user attributes, and financial transactions to provide a holistic view of each loan applicant.
- **Preprocessing:** 
  - **Principal Component Analysis (PCA):** Reduces dimensionality while retaining 95% of the variance.
  - **Scaling:** Standardizes features to ensure optimal performance of the SVM classifier.

---

## 📊 Visualization

- **`/figs`:** Contains charts and graphs that illustrate model performance metrics and insights derived from data analysis.
- **`/csv`:** Houses processed datasets that have undergone cleaning, feature engineering, and transformation for modeling purposes.

---

