from sentiment_analysis import transcribe_with_chunks
from google_sheets import store_data_in_sheet
from env_setup import config
from product_recommender import ProductRecommender
from objection_handler import ObjectionHandler, load_objections  # Ensure load_objections is imported
from sentence_transformers import SentenceTransformer

def main():
    # Load objections data
    objections_file_path = r"C:\Users\shaik\Downloads\Sales Calls Transcriptions - Sheet3.csv"
    objections_dict = load_objections(objections_file_path)

    # Initialize handlers and model
    product_recommender = ProductRecommender(r"C:\Users\shaik\Downloads\Sales Calls Transcriptions - Sheet2.csv")
    objection_handler = ObjectionHandler(objections_file_path)
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Call the transcription function
    transcribed_chunks = transcribe_with_chunks(objections_dict)

    total_text = ""
    sentiment_scores = []
    processed_chunks = []  # Store chunks with their real-time processing results

    for chunk, sentiment, score in transcribed_chunks:
        if chunk.strip():  
            total_text += chunk + " "
            sentiment_scores.append(score if sentiment == "POSITIVE" else -score)

            # Get embeddings for similarity check
            query_embedding = model.encode([chunk])
            
            # Only process recommendations if there's high similarity with products
            product_distances, _ = product_recommender.index.search(query_embedding, 1)
            if product_distances[0][0] < 1.5:  # Same threshold as real-time
                recommendations = product_recommender.get_recommendations(chunk)
                if recommendations:
                    print(f"Recommendations for chunk: '{chunk}'")
                    for idx, rec in enumerate(recommendations, 1):
                        print(f"{idx}. {rec}")
            
            # Only process objections if there's high similarity with objections
            objection_distances, _ = objection_handler.index.search(query_embedding, 1)
            if objection_distances[0][0] < 1.5:  # Same threshold as real-time
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
