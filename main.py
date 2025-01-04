from sentiment_analysis import transcribe_with_chunks
from google_sheets import store_data_in_sheet
from env_setup import config
from product_recommender import get_recommendations  # Import the recommendation function

def main():
    chunks = transcribe_with_chunks()  # Transcribing audio chunks
    total_text = ""
    sentiment_scores = []

    # Iterate over each chunk, its sentiment, and score
    for chunk, sentiment, score in chunks:
        if chunk.strip():  # Only process non-empty chunks
            total_text += chunk + " "  # Concatenate chunk to total_text
            sentiment_scores.append(score if sentiment == "POSITIVE" else -score)

            # Check if the chunk contains relevant keywords before getting recommendations
            if any(keyword in chunk.lower() for keyword in ["dress", "shoes", "clothes", "apparel", "fashion"]):
                recommendations = get_recommendations(chunk)
                if recommendations:
                    print(f"\nRecommendations for chunk: '{chunk}'")
                    for idx, rec in enumerate(recommendations, 1):
                        print(f"{idx}. {rec}")
                else:
                    print("No recommendations found.")
            else:
                print(f"\nNo relevant keywords found in the chunk: '{chunk}'. Skipping recommendations.")

            print(f"Chunk: '{chunk}' | Sentiment: {sentiment} | Score: {score}")

    # Calculate overall sentiment of the conversation
    overall_sentiment = "POSITIVE" if sum(sentiment_scores) > 0 else (
        "NEGATIVE" if sum(sentiment_scores) < 0 else "NEUTRAL"
    )

    print("\nConversation Summary:")
    print(total_text.strip())
    print(f"Overall Sentiment: {overall_sentiment}")

    # Store data in Google Sheets
    store_data_in_sheet(config["google_sheet_id"], chunks, total_text.strip(), overall_sentiment)

if __name__ == "__main__":
    main()
