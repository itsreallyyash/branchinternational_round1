import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

# Step 1: Load the dataset
file_path = 'csv/engineered_data.csv'
data = pd.read_csv(file_path)

# Step 2: Data Preparation
# Drop irrelevant columns and handle missing values
data_cleaned = data.drop(columns=['user_id', 'loan_outcome', 'location_provider'])
data_cleaned = data_cleaned.fillna(data_cleaned.median())

# Separate features and target
X = data_cleaned.drop(columns=['loan_outcome_encoded'])
y = data_cleaned['loan_outcome_encoded']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: PCA for Dimensionality Reduction
pca = PCA(n_components=0.95)  # Retain 95% variance
X_pca = pca.fit_transform(X_scaled)
print(f"PCA reduced dimensions from {X_scaled.shape[1]} to {X_pca.shape[1]}")

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42, stratify=y)

# Step 5: Model Training
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# Step 6: Model Evaluation
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Classification Report
report = classification_report(y_test, y_pred, target_names=["Not Defaulted", "Defaulted"])
print("Classification Report:\n", report)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)


# Step 1: Hyperparameter Tuning for Random Forest
rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
}

rf_grid = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=rf_params,
    scoring='accuracy',
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    verbose=1,
    n_jobs=-1
)
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_

# Step 2: Hyperparameter Tuning for SVM
svm_params = {
    'svc__C': [0.1, 1, 10],
    'svc__gamma': ['scale', 'auto'],
    'svc__kernel': ['linear', 'rbf']
}

svm_pipeline = Pipeline([('scaler', StandardScaler()), ('svc', SVC(probability=True))])
svm_grid = GridSearchCV(
    estimator=svm_pipeline,
    param_grid=svm_params,
    scoring='accuracy',
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    verbose=1,
    n_jobs=-1
)
svm_grid.fit(X_train, y_train)
best_svm = svm_grid.best_estimator_

# Step 3: Evaluate Both Models
rf_y_pred = best_rf.predict(X_test)
rf_y_proba = best_rf.predict_proba(X_test)[:, 1]
svm_y_pred = best_svm.predict(X_test)
svm_y_proba = best_svm.predict_proba(X_test)[:, 1]

# Classification Reports
rf_report = classification_report(y_test, rf_y_pred, target_names=["Not Defaulted", "Defaulted"])
svm_report = classification_report(y_test, svm_y_pred, target_names=["Not Defaulted", "Defaulted"])
print("Random Forest Classification Report:\n", rf_report)
print("SVM Classification Report:\n", svm_report)

# Confusion Matrices
rf_conf_matrix = confusion_matrix(y_test, rf_y_pred)
svm_conf_matrix = confusion_matrix(y_test, svm_y_pred)

# Step 4: Visualizations
# Random Forest Confusion Matrix
plt.figure(figsize=(6, 6))
plt.matshow(rf_conf_matrix, cmap='coolwarm', alpha=0.8)
plt.title("Random Forest Confusion Matrix", pad=20)
plt.colorbar()
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(ticks=[0, 1], labels=["Not Defaulted", "Defaulted"])
plt.yticks(ticks=[0, 1], labels=["Not Defaulted", "Defaulted"])
plt.show()

# SVM Confusion Matrix
plt.figure(figsize=(6, 6))
plt.matshow(svm_conf_matrix, cmap='coolwarm', alpha=0.8)
plt.title("SVM Confusion Matrix", pad=20)
plt.colorbar()
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(ticks=[0, 1], labels=["Not Defaulted", "Defaulted"])
plt.yticks(ticks=[0, 1], labels=["Not Defaulted", "Defaulted"])
plt.show()

# ROC Curves
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_y_proba)
rf_auc = auc(rf_fpr, rf_tpr)
svm_fpr, svm_tpr, _ = roc_curve(y_test, svm_y_proba)
svm_auc = auc(svm_fpr, svm_tpr)

plt.figure(figsize=(8, 6))
plt.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {rf_auc:.2f})')
plt.plot(svm_fpr, svm_tpr, label=f'SVM (AUC = {svm_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.show()

# PCA Explained Variance
plt.figure(figsize=(8, 6))
plt.bar(range(pca.n_components_), pca.explained_variance_ratio_)
plt.title("PCA Explained Variance Ratio")
plt.xlabel("Principal Components")
plt.ylabel("Explained Variance")
plt.show()



# Step 7: Visualizations
# Confusion Matrix Heatmap
plt.figure(figsize=(6, 6))
plt.matshow(conf_matrix, cmap='coolwarm', alpha=0.8)
plt.title("Confusion Matrix", pad=20)
plt.colorbar()
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(ticks=[0, 1], labels=["Not Defaulted", "Defaulted"])
plt.yticks(ticks=[0, 1], labels=["Not Defaulted", "Defaulted"])
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.show()

# PCA Explained Variance
plt.figure(figsize=(8, 6))
plt.bar(range(pca.n_components_), pca.explained_variance_ratio_)
plt.title("PCA Explained Variance Ratio")
plt.xlabel("Principal Components")
plt.ylabel("Explained Variance")
plt.show()
