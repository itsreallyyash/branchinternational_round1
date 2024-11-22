import pandas as pd

file_path = "combined_data.csv"  
df = pd.read_csv(file_path)
print("Dataset Info:")
print(df.info())

print("\nFirst Few Rows of the Dataset:")
print(df.head())
print("\nSummary Statistics:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

loan_outcome_counts = df['loan_outcome'].value_counts()
print("\nLoan Outcome Counts:")
print(loan_outcome_counts)

gps_summary = df[['longitude', 'latitude', 'accuracy']].describe()
print("\nGPS Data Summary:")
print(gps_summary)

df['application_at'] = pd.to_datetime(df['application_at'])
df['gps_fix_at'] = pd.to_datetime(df['gps_fix_at'])
df['server_upload_at'] = pd.to_datetime(df['server_upload_at'])

df['gps_upload_delay'] = (df['server_upload_at'] - df['gps_fix_at']).dt.total_seconds()

df['loan_outcome_encoded'] = df['loan_outcome'].apply(lambda x: 1 if x == 'repaid' else 0)

df.to_csv("processed_data.csv", index=False)
print("\nProcessed data saved as 'processed_data.csv'.")
