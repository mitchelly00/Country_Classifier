# spliting data 
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_pickle('tables_with_fixed_features.pkl')

print(len(df))
print(df['Feature'].head())


# Step 2: Split into train and temp (val + test), stratified
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df['Dance'],
    random_state=42,
    shuffle=True
)

print("train")

# Step 3: Split temp into val and test (50/50 of the 30%), still stratified
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df['Dance'],
    random_state=42,
    shuffle=True
)

print("val test")

#Test Balance in each
print("Train:")
print(train_df['Dance'].value_counts())
print("\nValidation:")
print(val_df['Dance'].value_counts())
print("\nTest:")
print(test_df['Dance'].value_counts())

#saving it all
file_path = 'tables_with_fixed_features.pkl'

# Save as pickle
df.to_pickle(file_path)


# #upload to S3
s3 = boto3.client('s3')

s3.upload_file(file_path, 'ucwdc-country-classifier', file_path)
print("Exported to S3")

