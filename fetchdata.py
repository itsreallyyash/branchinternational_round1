import psycopg2
import pandas as pd

db_config = {
    "host": "branchhomeworkdb.cv8nj4hg6yra.ap-south-1.rds.amazonaws.com",
    "port": 5432,
    "database": "branchdsprojectgps",
    "user": "datascientist",
    "password": "47eyYBLT0laW5j9U24Uuy8gLcrN"
}

try:
    conn = psycopg2.connect(**db_config)
    print("Database connection successful.")
except Exception as e:
    print(f"Error connecting to the database: {e}")
    exit()

queries = {
    "loan_outcomes": "SELECT * FROM loan_outcomes;",
    "gps_fixes": "SELECT * FROM gps_fixes;",
    "user_attributes": "SELECT * FROM user_attributes;"
}

try:
    loan_outcomes_df = pd.read_sql(queries["loan_outcomes"], conn)
    gps_fixes_df = pd.read_sql(queries["gps_fixes"], conn)
    user_attributes_df = pd.read_sql(queries["user_attributes"], conn)
    print("Tables loaded successfully into DataFrames.")
except Exception as e:
    print(f"Error loading tables: {e}")
    conn.close()
    exit()

try:
    combined_df = pd.merge(loan_outcomes_df, user_attributes_df, on="user_id")
    combined_df = pd.merge(combined_df, gps_fixes_df, on="user_id", how="left")
    print("Tables merged successfully.")
except Exception as e:
    print(f"Error merging tables: {e}")
    conn.close()
    exit()

loan_outcomes_df.to_csv("loan_outcomes.csv", index=False)
gps_fixes_df.to_csv("gps_fixes.csv", index=False)
user_attributes_df.to_csv("user_attributes.csv", index=False)
combined_df.to_csv("combined_data.csv", index=False)
print("Data saved.")
conn.close()
print(combined_df.head())
