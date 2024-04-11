
import os
import subprocess

# The ID of the folder to download
folder_id = "1awzQfYpkqx8KgRiuHJSYgHgNYAZuo76y"
# The local directory to save the downloaded folder
local_path = "models/"

def download_folder_from_google_drive(folder_id, local_path):
    """Downloads the specified folder from Google Drive using gdown."""
    command = [
        "gdown",
        "--folder",
        f"https://drive.google.com/drive/folders/{folder_id}",
        "-O",
        local_path,
        "--no-cookies"
    ]
    subprocess.check_call(command)

if __name__ == "__main__":
    # Create the local directory if it does not exist
    if not os.path.exists(local_path):
        os.makedirs(local_path)
    
    # Download the folder
    download_folder_from_google_drive(folder_id, local_path)
    print("Download completed successfully.")
