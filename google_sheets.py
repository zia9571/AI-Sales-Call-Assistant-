import uuid
from google.oauth2 import service_account
from googleapiclient.discovery import build
from env_setup import config

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
