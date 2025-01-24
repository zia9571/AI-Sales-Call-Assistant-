import uuid
from google.oauth2 import service_account
from googleapiclient.discovery import build
from env_setup import config
import pandas as pd

SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

def authenticate_google_account():
    service_account_file = config["google_creds"]
    if not service_account_file:
        raise ValueError("Service account credentials path is missing in env_setup.py.")
    return service_account.Credentials.from_service_account_file(service_account_file, scopes=SCOPES)

def store_data_in_sheet(sheet_id, chunks, summary, overall_sentiment):
    creds = authenticate_google_account()
    service = build('sheets', 'v4', credentials=creds)
    sheet = service.spreadsheets()

    call_id = str(uuid.uuid4())
    print(f"Call ID: {call_id}")

    values = []
    if chunks:
        first_chunk, first_sentiment, _ = chunks[0]
        values.append([call_id, first_chunk, first_sentiment, summary, overall_sentiment])
    for chunk, sentiment, _ in chunks[1:]:
        values.append(["", chunk, sentiment, "", ""])

    header = ["Call ID", "Chunk", "Sentiment", "Summary", "Overall Sentiment"]
    all_values = [header] + values

    body = {'values': all_values}
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

def fetch_call_data(sheet_id, sheet_range="Sheet1!A1:E"):
    """
    Fetches data from the specified Google Sheet and returns a pandas DataFrame.

    :param sheet_id: The ID of the Google Sheet to fetch data from.
    :param sheet_range: The range in A1 notation to fetch data from.
    :return: pandas DataFrame with the sheet data.
    """
    try:
        creds = authenticate_google_account()
        service = build('sheets', 'v4', credentials=creds)
        sheet = service.spreadsheets()

        result = sheet.values().get(
            spreadsheetId=sheet_id,
            range=sheet_range
        ).execute()
        
        rows = result.get("values", [])
        
        if rows:
            headers = rows[0]
            data = rows[1:]
            
            df = pd.DataFrame(data, columns=headers)
            
            return df
        else:
            return pd.DataFrame()
    
    except Exception as e:
        print(f"Error fetching data from Google Sheets: {e}")
        return pd.DataFrame()
