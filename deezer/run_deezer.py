import requests
import os
import boto3
import pandas as pd
import io

bucket_name = 'ucwdc-country-classifier'

def search_deezer_track(artist, track):
    query = f"{track} {artist}"
    url = f"https://api.deezer.com/search?q={requests.utils.quote(query)}&limit=1"
    res = requests.get(url)
    data = res.json()
    if data['data']:
        track_info = data['data'][0]
        return {
            "title": track_info['title'],
            "artist": track_info['artist']['name'],
            "album": track_info['album']['title'],
            "release_date": track_info.get('release_date', 'N/A'),
            "preview_url": track_info['preview'],  # 30-sec mp3 preview
            "deezer_link": track_info['link']
        }
    else:
        return None

def download_preview(preview_url, id, bucket_name='ucwdc-country-classifier', folder="./Previews"):
    # if not preview_url:
    #     print("No preview available to download.")
    #     return
    
    # # Make sure the folder exists
    # os.makedirs(folder, exist_ok=True)
    s3 = boto3.client('s3')


    # Define object key (i.e., path in bucket)
    s3_key = f"{folder}/{id}.mp3"
    
     # Fetch the MP3 file
    response = requests.get(preview_url)
    if response.status_code == 200:
        # Use in-memory buffer
        mp3_buffer = io.BytesIO(response.content)

        # Upload to S3
        s3.put_object(Bucket=bucket_name, Key=s3_key, Body=mp3_buffer)
    else:
        print(f"Failed to download preview: HTTP {response.status_code} {id}")
    
    
#Read csv
s3 = boto3.client('s3')
bucket = 'ucwdc-country-classifier'
key = 'combined_tables.csv'

obj = s3.get_object(Bucket=bucket, Key=key)
df = pd.read_csv(io.BytesIO(obj['Body'].read()))

list = []
df_head = df.head()
for index, row in df.iterrows():
    song_artist = row['Artist/Group']
    song_name = row["Song Title"]
    song_dance = row['Dance']
    info = search_deezer_track(song_artist, song_name)
    if info and info["preview_url"]:
        list.append("true")
        download_preview(info["preview_url"], folder=song_dance, id = index)
    else:
        list.append("false")

df['Found'] = list

# Export to CSV
file_path = './combined_tables_with_validation.csv'
df.to_csv(file_path, index=False, header=True)


#added to S3
object_name = 'combined_tables_with_validation.csv'  # S3 
s3.upload_file(file_path, bucket_name, object_name)
print("Exported to S3")

# # Example usage
# info = search_deezer_track(song_artist, song_name)

# if info and info["preview_url"]:

#     #download_preview(info["preview_url"], filename=f"{song_name}.mp3")

