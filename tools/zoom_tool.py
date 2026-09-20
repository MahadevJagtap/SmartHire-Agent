import os
import requests
import base64
from langchain_core.tools import tool

def get_zoom_access_token():
    """Retrieve access token using Server-to-Server OAuth."""
    account_id = os.getenv("ZOOM_ACCOUNT_ID", "").strip()
    client_id = os.getenv("ZOOM_CLIENT_ID", "").strip()
    client_secret = os.getenv("ZOOM_CLIENT_SECRET", "").strip()
    
    if not all([account_id, client_id, client_secret]):
        return None

    url = f"https://zoom.us/oauth/token?grant_type=account_credentials&account_id={account_id}"
    auth_header = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    
    headers = {
        "Authorization": f"Basic {auth_header}"
    }
    
    try:
        response = requests.post(url, headers=headers)
        response.raise_for_status()
        return response.json().get("access_token")
    except Exception as e:
        print(f"Error fetching Zoom token: {e}")
        return None

@tool
def zoom_meeting_tool(candidate_name: str) -> str:
    """
    Generates a real Zoom meeting link for the interview invitation.
    If API fails due to account locks, falls back to PERSONAL_ZOOM_LINK or a descriptive URL.
    """
    # Fallback to personal link if provided in .env
    personal_link = os.getenv("PERSONAL_ZOOM_LINK", "").strip()
    
    token = get_zoom_access_token()
    if not token:
        return personal_link if personal_link else f"https://zoom.us/simulation/credentials-missing"

    # 1. Fetch User ID
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    try:
        user_resp = requests.get("https://api.zoom.us/v2/users", headers=headers)
        if user_resp.status_code == 200:
            users = user_resp.json().get("users", [])
            user_id = users[0].get("id") if users else "me"
        else:
            user_id = "me"
    except:
        user_id = "me"

    # 2. Create the meeting with absolute minimal settings to avoid account lock conflicts
    url = f"https://api.zoom.us/v2/users/{user_id}/meetings"
    data = {
        "topic": f"Interview with {candidate_name}",
        "type": 2
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 201:
            return response.json().get("join_url")
        
        # If specific 400 error about locked settings occurs
        error_msg = response.text
        if "locked by your account administrator" in error_msg:
            print(f"Warning: Zoom JBH setting is locked. Use PERSONAL_ZOOM_LINK in .env as a workaround.")
            return personal_link if personal_link else f"https://zoom.us/simulation/locked-account-settings-see-env"
            
        print(f"Zoom API Error: {error_msg}")
        return personal_link if personal_link else f"https://zoom.us/simulation/api-error-{response.status_code}"
    except Exception as e:
        print(f"Zoom Exception: {e}")
        return personal_link if personal_link else f"https://zoom.us/simulation/exception-occurred"
