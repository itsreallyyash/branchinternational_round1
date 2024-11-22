import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def load_and_prepare_data(file_path, sample_fraction=0.6):
    """
    Load the dataset, clean it, and optionally sample a fraction of it.

    Parameters:
    - file_path: str, path to the CSV file.
    - sample_fraction: float, fraction of data to sample (between 0 and 1).

    Returns:
    - X: pandas DataFrame, features.
    - y: pandas Series, target variable.
    """
    logging.info("Loading dataset...")
    data = pd.read_csv(file_path)
    logging.info("Cleaning data...")
    # Drop irrelevant columns
    data_cleaned = data.drop(columns=['user_id', 'loan_outcome', 'location_provider'])
    # Handle missing values by filling with median
    data_cleaned = data_cleaned.fillna(data_cleaned.median())
    
    # Optionally sample a fraction of the data to speed up training
    if sample_fraction < 1.0:
        logging.info(f"Sampling {sample_fraction*100}% of the data for faster training...")
        data_cleaned = data_cleaned.sample(frac=sample_fraction, random_state=42)
        logging.info(f"Sampled data shape: {data_cleaned.shape}")
    
    # Separate features and target
    X = data_cleaned.drop(columns=['loan_outcome_encoded'])
    y = data_cleaned['loan_outcome_encoded']
    
    # Ensure labels are binary (0 and 1)
    unique_labels = y.unique()
    if not set(unique_labels).issubset({0, 1}):
        raise ValueError(f"Labels should be binary encoded as 0 and 1. Found labels: {unique_labels}")
    
    logging.info("Data preparation completed.")
    return X, y

def define_pipeline():
    """
    Define the machine learning pipeline and hyperparameter grid.

    Returns:
    - pipeline: sklearn Pipeline object.
    - param_grid: dict, parameter grid for GridSearchCV.
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95)),  # Retain 95% variance
        ('svc', SVC(probability=True, random_state=42))
    ])
    
    param_grid = {
        'svc__C': [0.1, 1, 10],
        'svc__gamma': ['scale', 'auto'],
        'svc__kernel': ['linear', 'rbf']
    }
    
    return pipeline, param_grid

def split_data(X, y, test_size=0.4):
    """
    Split the data into training and testing sets.

    Parameters:
    - X: pandas DataFrame, features.
    - y: pandas Series, target variable.
    - test_size: float, proportion of data to be used as test set.

    Returns:
    - X_train, X_test, y_train, y_test: split datasets.
    """
    logging.info("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    logging.info(f"Training set size: {X_train.shape[0]} samples")
    logging.info(f"Testing set size: {X_test.shape[0]} samples")
    return X_train, X_test, y_train, y_test

def hyperparameter_tuning(pipeline, param_grid, X_train, y_train):
    """
    Perform hyperparameter tuning using GridSearchCV.

    Parameters:
    - pipeline: sklearn Pipeline object.
    - param_grid: dict, parameter grid for GridSearchCV.
    - X_train: pandas DataFrame, training features.
    - y_train: pandas Series, training target.

    Returns:
    - best_estimator: sklearn Pipeline, best found model.
    """
    logging.info("Starting hyperparameter tuning for SVM...")
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='accuracy',
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        verbose=2,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    logging.info(f"Best parameters found: {grid_search.best_params_}")
    logging.info(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the test set.

    Parameters:
    - model: sklearn estimator, trained model.
    - X_test: pandas DataFrame, testing features.
    - y_test: pandas Series, testing target.
    """
    logging.info("Evaluating the model on the test set...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Classification Report
    report = classification_report(y_test, y_pred, target_names=["Not Defaulted", "Defaulted"])
    print("Classification Report:\n", report)
    
    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    # Visualizations
    # Confusion Matrix Heatmap
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="coolwarm",
                xticklabels=["Not Defaulted", "Defaulted"],
                yticklabels=["Not Defaulted", "Defaulted"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
    
    # ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'SVM (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.show()

def save_model(model, filename='svm_pipeline.joblib'):
    """
    Save the trained model to a file.

    Parameters:
    - model: sklearn estimator, trained model.
    - filename: str, filename to save the model.
    """
    logging.info(f"Saving the trained model to {filename}...")
    dump(model, filename)
    logging.info("Model saved successfully.")

# Main Execution Flow
if __name__ == "__main__":
    try:
        # Parameters
        FILE_PATH = 'csv/engineered_data.csv'
        SAMPLE_FRACTION = 0.6  # Use 60% of data for faster training
        TEST_SIZE = 0.4  # 40% for testing
        
        # Load and Prepare Data
        X, y = load_and_prepare_data(FILE_PATH, sample_fraction=SAMPLE_FRACTION)
        
        # Split Data
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=TEST_SIZE)
        
        # Define Pipeline and Parameter Grid
        pipeline, param_grid = define_pipeline()
        
        # Hyperparameter Tuning
        best_svm = hyperparameter_tuning(pipeline, param_grid, X_train, y_train)
        
        # Evaluate the Best Model
        evaluate_model(best_svm, X_test, y_test)
        
        # Save the Model
        save_model(best_svm, 'svm_pipeline.joblib')
        
    except Exception as e:
        logging.error(f"An error occurred: {e}")
