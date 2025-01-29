import os
from dotenv import load_dotenv

load_dotenv()

config = {
    "google_creds": os.getenv("google_creds"),
    "huggingface_api_key": os.getenv("huggingface_api_key"),
    "google_sheet_id": os.getenv("google_sheet_id"), 
    "vosk_model_path": os.getenv("vosk_model_path"),
    "PRODUCT_DATA_PATH": os.getenv("PRODUCT_DATA_PATH"),
    "OBJECTION_DATA_PATH": os.getenv("OBJECTION_DATA_PATH"),
    "cohere_api_key": os.getenv("cohere_api_key")
}
