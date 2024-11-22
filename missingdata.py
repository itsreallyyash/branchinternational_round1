import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset
file_path = 'processed_data.csv'  # Update with your file path if necessary
df = pd.read_csv(file_path)

# 1. Remove rows with missing values
print("Original Dataset Shape:", df.shape)
df = df.dropna()
print("Dataset Shape After Dropping Missing Values:", df.shape)

# # 2. Handle Outliers (for GPS accuracy and other fields if necessary)
# # Removing rows with unreasonable GPS accuracy values
# df = df[df['accuracy'] > 0]
# df = df[df['accuracy'] < 5000]  # Adjust threshold as per domain knowledge
# print("Dataset Shape After Removing Outliers in 'accuracy':", df.shape)

# 3. Encode Categorical Features
# Encode 'loan_outcome' (repaid -> 1, defaulted -> 0)
label_encoder = LabelEncoder()
df['loan_outcome_encoded'] = label_encoder.fit_transform(df['loan_outcome'])

# 4. Feature Engineering
# Calculate the time delay between 'gps_fix_at' and 'server_upload_at'
df['gps_fix_at'] = pd.to_datetime(df['gps_fix_at'])
df['server_upload_at'] = pd.to_datetime(df['server_upload_at'])
df['gps_upload_delay'] = (df['server_upload_at'] - df['gps_fix_at']).dt.total_seconds()

# Drop unnecessary columns (like original timestamp fields if not needed)
df = df.drop(['gps_fix_at', 'server_upload_at'], axis=1)

# 5. Save the Cleaned Dataset
cleaned_file_path = 'cleaned_data.csv'
df.to_csv(cleaned_file_path, index=False)
print(f"Cleaned dataset saved to {cleaned_file_path}.")

# 6. Basic Exploratory Data Analysis (EDA)
# Visualization: Cash incoming vs Loan Outcome
sns.boxplot(data=df, x='loan_outcome', y='cash_incoming_30days')
plt.title("Cash Incoming vs Loan Outcome")
plt.show()

# Visualization: Age distribution by Loan Outcome
sns.histplot(data=df, x='age', kde=True, hue='loan_outcome')
plt.title("Age Distribution by Loan Outcome")
plt.show()

# Visualization: GPS upload delay distribution
sns.histplot(data=df, x='gps_upload_delay', kde=True, hue='loan_outcome')
plt.title("GPS Upload Delay Distribution by Loan Outcome")
plt.show()

# 7. Prepare Data for Modeling
# Selecting features and target
X = df[['age', 'cash_incoming_30days', 'gps_upload_delay', 'longitude', 'latitude', 'accuracy']]
y = df['loan_outcome_encoded']

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training Set Shape: {X_train.shape}, Test Set Shape: {X_test.shape}")

# Save training and testing sets for model training
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)
print("Training and testing sets saved as CSV files.")
