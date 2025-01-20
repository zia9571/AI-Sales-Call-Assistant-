from sentiment_analysis import transcribe_with_chunks
from google_sheets import store_data_in_sheet
from env_setup import config
from product_recommender import ProductRecommender
from objection_handler import load_objections, ObjectionHandler

def main():
    # Load objections and products
    product_recommender = ProductRecommender(r"C:\Users\shaik\Downloads\Sales Calls Transcriptions - Sheet2.csv")
    objection_handler = ObjectionHandler(r"C:\Users\shaik\Downloads\Sales Calls Transcriptions - Sheet3.csv")

    # Load objections at the start of the script
    objections_file_path = r"C:\Users\shaik\Downloads\Sales Calls Transcriptions - Sheet3.csv"
    objections_dict = load_objections(objections_file_path)

    # Call the transcription function which now includes objection handling
    transcribed_chunks = transcribe_with_chunks(objections_dict)

    total_text = ""
    sentiment_scores = []

    for chunk, sentiment, score in transcribed_chunks:
        if chunk.strip():  
            total_text += chunk + " "  # Accumulate the conversation text
            sentiment_scores.append(score if sentiment == "POSITIVE" else -score)

            # Check for product recommendations
            recommendations = product_recommender.get_recommendations(chunk)
            if recommendations:
                print(f"Recommendations for chunk: '{chunk}'")
                for idx, rec in enumerate(recommendations, 1):
                    print(f"{idx}. {rec}")

            # Check for objections
            objection_responses = objection_handler.handle_objection(chunk)
            if objection_responses:
                for response in objection_responses:
                    print(f"Objection Response: {response}")

    # Determine overall sentiment
    overall_sentiment = "POSITIVE" if sum(sentiment_scores) > 0 else "NEGATIVE"
    print(f"Overall Sentiment: {overall_sentiment}")

    # Generate a summary of the conversation
    print(f"Conversation Summary: {total_text.strip()}")

    # Store data in Google Sheets
    store_data_in_sheet(config["google_sheet_id"], transcribed_chunks, total_text.strip(), overall_sentiment)

if __name__ == "__main__":
    main()
