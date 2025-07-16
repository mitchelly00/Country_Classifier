import requests
import os
from flask import Flask, request
from dotenv import load_dotenv, set_key
from pathlib import Path
import os

load_dotenv()  # Loads variables from .env into environment
env_path = Path('.') / '.env'

CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
REDIRECT_URI = os.getenv("REDIRECT_URI")

app = Flask(__name__)


@app.route('/callback')
def callback():
    code = request.args.get('code')

    if not code:
        return "No code found in callback."

    # Step 3: Exchange code for token
    token_url = 'https://accounts.spotify.com/api/token'
    payload = {
        'grant_type': 'authorization_code',
        'code': code,
        'redirect_uri': REDIRECT_URI,
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
    }

    response = requests.post(token_url, data=payload)
    token_info = response.json()

    #Refresh Token 
    if 'refresh_token' in token_info:
        refresh_token = token_info['refresh_token']
        # Save it to the .env file
        set_key(env_path, "REFRESH_TOKEN", refresh_token)

    access_token = token_info.get('access_token')

    if not access_token:
        return f"Failed to get token: {token_info}"

    # Step 4: Use access token
    headers = {
        'Authorization': f'Bearer {access_token}'
    }

    me = requests.get("https://api.spotify.com/v1/me", headers=headers).json()

    return f"Hello, {me['display_name']}! Your Spotify ID is {me['id']}."

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000)


