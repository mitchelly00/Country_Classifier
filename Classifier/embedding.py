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

class STFTPatched(STFT):
    pass

def mp3_to_wav_bytes(mp3_bytes: bytes) -> io.BytesIO:
    """
    Convert mp3 bytes to wav bytes in memory using ffmpeg.
    """
    ffmpeg_cmd = [
        "ffmpeg",
        "-i", "pipe:0",      # input from stdin
        "-f", "wav",
        "pipe:1"             # output to stdout
    ]
    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    wav_bytes, err = process.communicate(input=mp3_bytes)
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg error: {err.decode()}")
    return io.BytesIO(wav_bytes)

def extract_row_embedding(key):
    global model
    try: 
        return extract_openl3_embedding_from_s3( key, model)
    except Exception as e:
        print(f"Failed to process {key}: {e}")
        return None  # or np.nan

def extract_openl3_embedding_from_s3(key: str, model=None) -> np.ndarray:
    """
    Downloads mp3 from S3, converts to WAV, extracts OpenL3 embedding,
    and returns the averaged embedding vector.
    """
    bucket='ucwdc-country-classifier'
    s3 = boto3.client("s3")
    response = s3.get_object(Bucket=bucket, Key=key)
    mp3_bytes = response['Body'].read()

    # Convert MP3 bytes to WAV bytes
    wav_io = mp3_to_wav_bytes(mp3_bytes)
    wav_io.seek(0)

    # Load audio with soundfile
    audio, sr = sf.read(wav_io)

    # Load model once if not provided

    # Extract embeddings
    emb, ts = openl3.get_audio_embedding(audio, sr, model=model, verbose=0)

    # Aggregate embeddings by mean over time frames (axis 0)
    embedding_vector = emb.mean(axis=0)

    return embedding_vector

# Example usage
if __name__ == "__main__":
    # Pull file from s3
    bucket_name = "ucwdc-country-classifier"
    key = 'combined_tables_with_features_id.csv'
    s3 = boto3.client('s3')

    obj = s3.get_object(Bucket='ucwdc-country-classifier', Key=key)
    df = pd.read_csv(io.BytesIO(obj['Body'].read()))
    #df = df.head(5).copy()

    # Load model once for multiple calls if needed
    model = openl3.models.load_audio_embedding_model(input_repr="mel256",
                                                    content_type="music",
                                                    embedding_size=512)
    # Extract embeddings sequentially
    embeddings = []
    for k in tqdm(df["Key"].tolist(), desc="Extracting Embeddings"):
        try:
            emb = extract_openl3_embedding_from_s3(k, model)
        except Exception as e:
            print(f"Failed to process {k}: {e}")
            emb = None
        embeddings.append(emb)

    df["Embedding"] = embeddings

    file_path = './combined_tables_with_embedding.pkl'
    df.to_pickle(file_path)


    #added to S3
    object_name = 'combined_tables_with_embedding.pkl'  # S3 
    s3.upload_file(file_path,'ucwdc-country-classifier', object_name)
    print("Exported to S3")
   

    

    

