import pandas as pd
import numpy as np
from datetime import datetime

# Load the dataset
file_path = 'processed_data.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Handle missing values: Drop rows with missing GPS data for simplicity
data_cleaned = data.dropna(subset=['gps_fix_at', 'longitude', 'latitude'])

# Convert application_at to datetime
data_cleaned['application_at'] = pd.to_datetime(data_cleaned['application_at'], errors='coerce', format='mixed')

# Feature Engineering: Time-based features
data_cleaned['application_hour'] = data_cleaned['application_at'].dt.hour
data_cleaned['application_dayofweek'] = data_cleaned['application_at'].dt.dayofweek

# Feature Engineering: Financial features
data_cleaned['cash_incoming_per_day'] = data_cleaned['cash_incoming_30days'] / 30

# Haversine distance calculation function
def haversine(lon1, lat1, lon2, lat2):
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r

# Sort data by user_id and gps_fix_at
data_cleaned.sort_values(by=['user_id', 'gps_fix_at'], inplace=True)

# Shift longitude and latitude for distance calculation
data_cleaned['prev_longitude'] = data_cleaned.groupby('user_id')['longitude'].shift()
data_cleaned['prev_latitude'] = data_cleaned.groupby('user_id')['latitude'].shift()

# Calculate distance traveled
data_cleaned['distance_traveled'] = haversine(
    data_cleaned['longitude'], data_cleaned['latitude'],
    data_cleaned['prev_longitude'], data_cleaned['prev_latitude']
)

# Fill missing distances with 0 for first records per user
data_cleaned['distance_traveled'].fillna(0, inplace=True)

# Aggregate movement features per user
movement_features = data_cleaned.groupby('user_id')['distance_traveled'].agg(
    total_distance='sum',
    mean_distance='mean',
    max_distance='max'
).reset_index()

# Merge movement features back into the main dataset
data_cleaned = data_cleaned.merge(movement_features, on='user_id', how='left')

# Drop unnecessary columns for simplicity
data_cleaned = data_cleaned.drop(
    ['application_at', 'gps_fix_at', 'server_upload_at', 'prev_longitude', 'prev_latitude'], axis=1
)

# Save or display the cleaned and engineered dataset
data_cleaned.to_csv('engineered_data.csv', index=False)  # Save to CSV
print("Feature engineering complete! Dataset saved as 'engineered_data.csv'")
