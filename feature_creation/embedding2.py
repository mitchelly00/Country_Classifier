# this file tries to use parralle processing but it does not 

import boto3
import io
import subprocess
import soundfile as sf
import openl3
import numpy as np
import pandas as pd
from tqdm import tqdm
import kapre
from kapre.time_frequency import STFT
from keras.saving import register_keras_serializable
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
import multiprocessing

class STFTPatched(STFT):
    pass

# Global model (loaded per subprocess)
_model = None

def init_worker():
    global _model
    _model = openl3.models.load_audio_embedding_model(
        input_repr="mel256",
        content_type="music",
        embedding_size=512
    )

def mp3_to_wav_bytes(mp3_bytes: bytes) -> io.BytesIO:
    ffmpeg_cmd = [
        "ffmpeg",
        "-i", "pipe:0",
        "-f", "wav",
        "pipe:1"
    ]
    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    wav_bytes, err = process.communicate(input=mp3_bytes)
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg error: {err.decode()}")
    return io.BytesIO(wav_bytes)

def extract_openl3_embedding_from_s3(key: str) -> np.ndarray:
    s3 = boto3.client("s3")
    bucket = 'ucwdc-country-classifier'
    response = s3.get_object(Bucket=bucket, Key=key)
    mp3_bytes = response['Body'].read()

    wav_io = mp3_to_wav_bytes(mp3_bytes)
    wav_io.seek(0)
    audio, sr = sf.read(wav_io)

    # Use model loaded in init_worker
    global _model
    emb, ts = openl3.get_audio_embedding(audio, sr, model=_model, verbose=0)
    return emb.mean(axis=0)

def process_key(key):
    try:
        emb = extract_openl3_embedding_from_s3(key)
        return (key, emb)
    except Exception as e:
        print(f"Failed to process {key}: {e}")
        return (key, None)

if __name__ == "__main__":
    # Download input CSV from S3
    s3 = boto3.client('s3')
    key = 'combined_tables_with_features_id.csv'
    obj = s3.get_object(Bucket='ucwdc-country-classifier', Key=key)
    df = pd.read_csv(io.BytesIO(obj['Body'].read()))
    df = df.head(5).copy()  # TEMP: You can remove or adjust this

    keys = df["Key"].tolist()

    embeddings_map = {}
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count(), initializer=init_worker) as executor:
        futures = {executor.submit(process_key, k): k for k in keys}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting Embeddings"):
            k, emb = future.result()
            embeddings_map[k] = emb

    df["Embedding"] = df["Key"].map(embeddings_map)

    # Save locally and upload to S3
    file_path = './combined_tables_with_embedding.pkl'
    df.to_pickle(file_path)
    s3.upload_file(file_path, 'ucwdc-country-classifier', 'combined_tables_with_embedding.pkl')
    print("Exported to S3")

   

    

    

