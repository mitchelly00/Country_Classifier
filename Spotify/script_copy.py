import requests
import base64
import os
from dotenv import load_dotenv

load_dotenv()

client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")

def get_access_token():
    auth_str = f"{client_id}:{client_secret}"
    b64_auth_str = base64.b64encode(auth_str.encode()).decode()

    headers = {
    "Content-Type": "application/x-www-form-urlencoded"
    }

    data = {
    "grant_type": "client_credentials",
    "client_id": client_id,
    "client_secret": client_secret
    }

    response = requests.post("https://accounts.spotify.com/api/token", headers=headers, data=data)
    return response.json()["access_token"]
    

def search_track(song_name, artist_name, token):
    headers = {
        "Authorization": f"Bearer {token}"
    }

    query = f"{song_name} artist:{artist_name}"
    url = f"https://api.spotify.com/v1/search?q={requests.utils.quote(query)}&type=track&limit=1"

    response = requests.get(url, headers=headers)
    return response.json()

def extract_metadata(track_json):
    if track_json['tracks']['items']:
        track = track_json['tracks']['items'][0]
        return {
            "name": track["name"],
            "artist": track["artists"][0]["name"],
            "album": track["album"]["name"],
            "release_date": track["album"]["release_date"],
            "preview_url": track["preview_url"],  # This is the 30s mp3 clip
            "external_url": track["external_urls"]["spotify"]
        }
    else:
        return None
    

def download_preview(preview_url, filename="preview.mp3"):
    if preview_url:
        response = requests.get(preview_url)
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"Preview saved as {filename}")
    else:
        print("No preview available.")

if __name__ == "__main__":
    token = get_access_token()
    track_json = search_track("Man! I feel like a woman!", "Shania Twain", token)
    metadata = extract_metadata(track_json)
    print(metadata)
    download_preview(metadata["preview_url"])

