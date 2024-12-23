import uuid  # Ensure this is imported for generating unique Call IDs
import datetime
from google.oauth2 import service_account
from googleapiclient.discovery import build
from env_setup import config  # Ensure your env_setup.py has the necessary variables

SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

def authenticate_google_account():
    """
    Authenticate Google service account using the credentials JSON file.
    Returns the authenticated credentials object.
    """
    service_account_file = config.get("google_creds") 
    if not service_account_file:
        raise ValueError("Service account credentials path is missing in env_setup.py.")

    creds = service_account.Credentials.from_service_account_file(
        service_account_file, scopes=SCOPES
    )
    return creds

def store_data_in_sheet(sheet_id, chunks, summary, overall_sentiment):
    """
    Store transcription data in Google Sheets.
    :param sheet_id: ID of the Google Sheet
    :param chunks: List of tuples containing (chunk, sentiment, score)
    :param summary: Conversation summary
    :param overall_sentiment: Overall sentiment of the conversation
    """
    creds = authenticate_google_account()
    service = build('sheets', 'v4', credentials=creds)
    sheet = service.spreadsheets()

    # a unique Call ID
    call_id = str(uuid.uuid4())
    print(f"Call ID: {call_id}")

    # data for Google Sheets
    values = []
    
    # Adding the first row with Call ID, Summary, and Overall Sentiment
    if chunks:
        first_chunk, first_sentiment, _ = chunks[0]
        values.append([call_id, first_chunk, first_sentiment, summary, overall_sentiment])
    
    # Adding the remaining chunks with empty Call ID, Summary, and Overall Sentiment
    for chunk, sentiment, _ in chunks[1:]:
        values.append(["", chunk, sentiment, "", ""])

    # insert a header row if it's the first write (modify the range if necessary)
    header = ["Call ID", "Chunk", "Sentiment", "Summary", "Overall Sentiment"]
    all_values = [header] + values

    # write data to Google Sheets
    body = {
        'values': all_values
    }

    try:
        result = sheet.values().append(
            spreadsheetId=sheet_id,
            range="Sheet1!A1",  
            valueInputOption="RAW",
            body=body
        ).execute()

        print(f"{result.get('updates').get('updatedCells')} cells updated in Google Sheets.")
    except Exception as e:
        print(f"Error updating Google Sheets: {e}")
