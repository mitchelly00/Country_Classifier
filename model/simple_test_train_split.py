# spliting data 
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import boto3

df = pd.read_pickle('tables_with_combined_features.pkl')

print(len(df))
print(df['Feature'].head())

print(df.columns)


# Step 2: Split into train and temp (val + test), stratified
train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    stratify=df['Dance'],
    random_state=42,
    shuffle=True
)

print("train")


#Test Balance in each
print("Train:")
print(train_df['Dance'].value_counts())
print("\nTest:")
print(test_df['Dance'].value_counts())

#saving it all
# Create local directory if it doesn't exist
local_dir = "./data_splits"

# File names
file_map = {
    "train2.pkl": train_df,
    "test2.pkl": test_df,
}

# Save locally
for filename, df in file_map.items():
    path = os.path.join(local_dir, filename)
    df.to_pickle(path)
    print(f"Saved {filename} locally.")

# S3 upload
bucket = "ucwdc-country-classifier" 
s3_prefix = "data_splits/"
s3 = boto3.client("s3")

for filename in file_map:
    local_path = os.path.join(local_dir, filename)
    s3_key = s3_prefix + filename
    s3.upload_file(local_path, bucket, s3_key)
    print(f"Uploaded {filename} to s3://{bucket}/{s3_key}")

