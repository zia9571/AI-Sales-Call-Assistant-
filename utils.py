import os
from dotenv import load_dotenv

# function to load environment variables from .env file
def load_env_variables():
    load_dotenv()

    return {
        'huggingface_api_key': os.getenv('HUGGINGFACE_API_KEY'),
        'google_creds': os.getenv('GOOGLE_CREDS_PATH'),
        'google_sheet_id': os.getenv('GOOGLE_SHEET_ID')
    }
