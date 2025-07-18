import boto3
import librosa
import numpy as np
import pandas as pd
import io

def extract_features_from_s3(key, bucket='ucwdc-country-classifier'):
    s3 = boto3.client('s3')

    # Download mp3 file into memory
    response = s3.get_object(Bucket=bucket, Key=key)
    audio_bytes = io.BytesIO(response['Body'].read())

    # Load audio with librosa
    y, sr = librosa.load(audio_bytes, sr=None)

    # 1. MFCC (13)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    # 2. Chroma (12)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    # 3. Spectral Contrast (7)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = np.mean(contrast, axis=1)

    # 4. Zero Crossing Rate (1)
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr)

    # 5. Tempo (1)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    # Combine all features into one vector
    features = np.hstack([mfcc_mean, chroma_mean, contrast_mean, zcr_mean, tempo])
    print(features)
    return features




#added to S3

key = 'combined_tables_with_validation.csv'
s3 = boto3.client('s3')

obj = s3.get_object(Bucket='ucwdc-country-classifier', Key=key)
df = pd.read_csv(io.BytesIO(obj['Body'].read()))
df_found = df[df['Found']==True]


for index, row in df_found.head().iterrows():
    song_dance = row['Dance']
    song_key = f"{song_dance}/{index}.mp3"
    extract_features_from_s3(song_key)


#extract_features_from_s3()