from sentiment_analysis import transcribe_with_chunks
from google_sheets import store_data_in_sheet
from env_setup import config

def main():
    chunks = transcribe_with_chunks()
    total_text = ""
    sentiment_scores = []

    for chunk, sentiment, score in chunks:
        total_text += chunk + " "
        sentiment_scores.append(score if sentiment == "POSITIVE" else -score)

    overall_sentiment = "POSITIVE" if sum(sentiment_scores) > 0 else (
        "NEGATIVE" if sum(sentiment_scores) < 0 else "NEUTRAL"
    )

    print("\nConversation Summary:")
    print(total_text.strip())
    print(f"Overall Sentiment: {overall_sentiment}")

    store_data_in_sheet(config["google_sheet_id"], chunks, total_text.strip(), overall_sentiment)

if __name__ == "__main__":
    main()
