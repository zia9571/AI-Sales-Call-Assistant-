from sentiment_analysis import transcribe_with_chunks
from google_sheets import store_data_in_sheet
from env_setup import config
from product_recommender import get_recommendations
from objection_handler import load_objections
from google_sheets import store_data_in_sheet

def main():
    # Load objections at the start of the script
    objections_file_path = r"C:\Users\shaik\Downloads\Sales Calls Transcriptions - Sheet3.csv"
    objections_dict = load_objections(objections_file_path)

    # Call the transcription function which now includes objection handling
    transcribed_chunks = transcribe_with_chunks(objections_dict)

    total_text = ""
    sentiment_scores = []

    for chunk, sentiment, score in transcribed_chunks:
        if chunk.strip():  
            total_text += chunk + " "  
            sentiment_scores.append(score if sentiment == "POSITIVE" else -score)

            # Check for relevant keywords using the updated list
            if any(keyword in chunk.lower() for keyword in [
                "dress", "floral dress", "maxi dress", "velvet dress", "sundress",
                "shirt", "casual shirt", "formal shirt", "t-shirt", "pajama set", "blouse",
                "trousers", "jeans", "chinos", "joggers", "sweatpants",
                "skirt", "denim skirt", "maxi skirt", "pencil skirt",
                "shoes", "running shoes", "high heels", "loafers", "sandals", "boots", "sneakers",
                "socks", "luggage", "bag"  # Updated keyword list
            ]):
                print(f"Recommendations for chunk: '{chunk}'")
                recommendations = get_recommendations(chunk)
                for idx, rec in enumerate(recommendations, 1):
                    print(f"{idx}. {rec}")

    overall_sentiment = "POSITIVE" if sum(sentiment_scores) > 0 else "NEGATIVE"
    print(f"Conversation Summary: {total_text.strip()}")
    print(f"Overall Sentiment: {overall_sentiment}")

    # Store data in Google Sheets
    store_data_in_sheet(config["google_sheet_id"], transcribed_chunks, total_text.strip(), overall_sentiment)

if __name__ == "__main__":
    main()
