import os
from dotenv import load_dotenv


load_dotenv()
config = {
    "google_creds": os.getenv("google_creds"),
    "huggingface_api_key": os.getenv("huggingface_api_key"),
    "google_sheet_id": os.getenv("google_sheet_id"),  
    "tf": os.getenv("TF_ENABLE_ONEDNN_OPTS"),
}
