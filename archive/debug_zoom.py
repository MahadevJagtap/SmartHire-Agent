import os
import requests
import base64
from dotenv import load_dotenv

load_dotenv()

def get_zoom_access_token():
    """Retrieve access token using Server-to-Server OAuth."""
    account_id = os.getenv("ZOOM_ACCOUNT_ID", "").strip()
    client_id = os.getenv("ZOOM_CLIENT_ID", "").strip()
    client_secret = os.getenv("ZOOM_CLIENT_SECRET", "").strip()
    
    if not all([account_id, client_id, client_secret]):
        print("Missing credentials!")
        return None

    url = f"https://zoom.us/oauth/token?grant_type=account_credentials&account_id={account_id}"
    auth_header = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    
    headers = {
        "Authorization": f"Basic {auth_header}"
    }
    
    try:
        response = requests.post(url, headers=headers)
        if response.status_code != 200:
            print(f"Token Error {response.status_code}: {response.text}")
            return None
        return response.json().get("access_token")
    except Exception as e:
        print(f"Token Exception: {e}")
        return None

def test_zoom():
    token = get_zoom_access_token()
    if not token:
        print("Failed to get token.")
        return

    print("Success: Got Token")
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    # 1. Get User ID
    print("Fetching user list...")
    user_id = "me"
    try:
        user_resp = requests.get("https://api.zoom.us/v2/users", headers=headers)
        if user_resp.status_code == 200:
            users = user_resp.json().get("users", [])
            if users:
                user_id = users[0].get("id")
                print(f"Using User ID: {user_id}")
            else:
                print("No users found.")
        else:
            print(f"Error fetching users: {user_resp.text}")
    except Exception as e:
        print(f"User List Exception: {e}")

    # 2. Get User Settings
    print(f"Fetching settings for {user_id}...")
    try:
        sett_resp = requests.get(f"https://api.zoom.us/v2/users/{user_id}/settings", headers=headers)
        if sett_resp.status_code == 200:
            import json
            with open('zoom_settings.json', 'w') as f:
                json.dump(sett_resp.json(), f, indent=2)
            print("Dumped settings to zoom_settings.json")
        else:
            print(f"Error fetching settings ({sett_resp.status_code}): {sett_resp.text}")
    except Exception as e:
        print(f"Settings Exception: {e}")

    # 3. Create meeting
    url = f"https://api.zoom.us/v2/users/{user_id}/meetings"
    data = {
        "topic": "Recruitment Interview Test",
        "settings": {
            "join_before_host": True,
            "jbh_time": 0
        }
    }
    
    try:
        print(f"Creating meeting for {user_id}...")
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 201:
            print(f"Success! URL: {response.json().get('join_url')}")
        else:
            print(f"Meeting Error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"Meeting Exception: {e}")

if __name__ == "__main__":
    test_zoom()
