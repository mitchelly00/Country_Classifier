from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import boto3

def combine_features_normalized(df):
    # Convert lists/arrays to 2D numpy arrays
    feature_array = np.vstack(df['Feature'].values)
    embedding_array = np.vstack(df['Embedding'].values)
    
    # Normalize each separately
    scaler_feat = StandardScaler()
    scaler_emb = StandardScaler()

    feature_norm = scaler_feat.fit_transform(feature_array)
    embedding_norm = scaler_emb.fit_transform(embedding_array)
    
    # Combine normalized features
    combined = np.hstack([feature_norm, embedding_norm])
    
    # Put back to DataFrame
    df['Combined'] = list(combined)
    return df

df = pd.read_pickle('tables_with_fixed_features.pkl')

new_df = combine_features_normalized(df)

#saving file

file_path = 'tables_with_combined_features.pkl'

# Save as pickle
df.to_pickle(file_path)

# #upload to S3
s3 = boto3.client('s3')

s3.upload_file(file_path, 'ucwdc-country-classifier', file_path)
print("Exported to S3")