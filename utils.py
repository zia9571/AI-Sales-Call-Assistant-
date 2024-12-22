import os
from dotenv import load_dotenv

# Function to load environment variables from .env file
def load_env_variables():
    # Load environment variables from .env file
    load_dotenv()

    # Return the loaded environment variables as a dictionary
    return {
        'assemblyai_api_key': os.getenv('ASSEMBLYAI_API_KEY'),
        'huggingface_api_key': os.getenv('HUGGINGFACE_API_KEY'),
        'google_creds': os.getenv('GOOGLE_CREDS_PATH'),
        'google_sheet_id': os.getenv('GOOGLE_SHEET_ID')
    }
