import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Initialize the config dictionary to store environment variables
config = {
    "google_creds": os.getenv("google_creds"),
    "huggingface_api_key": os.getenv("huggingface_api_key"),
    "google_sheet_id": os.getenv("google_sheet_id"),  # Add this line
    "tf": os.getenv("TF_ENABLE_ONEDNN_OPTS"),
}
