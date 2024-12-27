import os
from dotenv import load_dotenv

load_dotenv()

config = {
    "google_creds": os.getenv("google_creds"),
    "huggingface_api_key": os.getenv("huggingface_api_key"),
    "google_sheet_id": os.getenv("google_sheet_id"), 
    "vosk_model_path": os.getenv("vosk_model_path")
}
