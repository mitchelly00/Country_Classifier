import boto3
import pandas as pd
import io




#added to S3

key = 'combined_tables_with_validation.csv'
s3 = boto3.client('s3')

obj = s3.get_object(Bucket='ucwdc-country-classifier', Key=key)
df = pd.read_csv(io.BytesIO(obj['Body'].read()))
df_found = df[df['Found']==True]

feature_list = []

for index, row in df_found.iterrows():
    song_dance = row['Dance']
    song_key = f"{song_dance}/{index}.mp3"
    feature_list.append(song_key)


#pulling tables with feature
key_feature = 'combined_tables_with_features.csv'
obj = s3.get_object(Bucket='ucwdc-country-classifier', Key=key_feature)
df_feature = pd.read_csv(io.BytesIO(obj['Body'].read()))

df_feature["Key"] = feature_list


#Saving locally
file_path = './combined_tables_with_features_id.csv'
df_feature.to_csv(file_path, index=False, header=True)


#added to S3
object_name = 'combined_tables_with_features_id.csv'  # S3 
s3.upload_file(file_path,'ucwdc-country-classifier', object_name)
print("Exported to S3")
print(df_feature.head())