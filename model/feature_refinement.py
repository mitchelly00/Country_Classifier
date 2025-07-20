import numpy as np; print(np.__version__)
import pandas as pd; print(pd.__version__)
import boto3
import io


# Load CSV
#df = pd.read_pickle('combined_tables_with_embedding.pkl')

bucket='ucwdc-country-classifier'
s3 = boto3.client("s3")
key = "combined_tables_with_embedding.pkl"
response = s3.get_object(Bucket=bucket, Key=key)

df = pd.read_pickle(io.BytesIO(response['Body'].read()))

print(len(df))
print(df['Feature'].head())


# # Convert stringified lists back to real Python lists
# df['Feature'] = df['Feature'].apply(lambda x: np.fromstring(x.strip("[]"), sep=' '))

# # Convert to numpy arrays
# print(type(df['Feature'].iloc[0,]))

# file_path = 'tables_with_fixed_features.pkl'

# # Save as pickle
# df.to_pickle(file_path)


# # #upload to S3
# s3 = boto3.client('s3')

# s3.upload_file(file_path, 'ucwdc-country-classifier', file_path)
# print("Exported to S3")
