import pandas as pd
import string

def load_objections(file_path):
    """Load objections and responses from a CSV file into a dictionary."""
    objections_df = pd.read_csv(r"C:\Users\shaik\Downloads\Sales Calls Transcriptions - Sheet3.csv")
    objections_dict = dict(zip(objections_df['Customer Objection'], objections_df['Salesperson Response']))
    return objections_dict

def normalize_text(text):
    """Normalize the text by converting to lowercase and removing punctuation."""
    return text.lower().translate(str.maketrans('', '', string.punctuation))

def check_objections(text, objections_dict):
    """Check if any objections are present in the text and return the corresponding response."""
    responses = []
    normalized_text = normalize_text(text)  # Normalize the transcription text

    # Check for specific keywords related to budget concerns
    budget_keywords = ["budget", "cost", "price", "expensive", "too much", "afford", "financial"]
    if any(keyword in normalized_text for keyword in budget_keywords):
        responses.append(("I don't have the budget for this", "I understand. Would it help if we could break the payment into more manageable installments?"))

    # Check against the objections in the dictionary
    for objection, response in objections_dict.items():
        normalized_objection = normalize_text(objection)  # Normalize the objection
        if normalized_objection in normalized_text:  # Check for the objection in normalized text
            responses.append((objection, response))

    return responses