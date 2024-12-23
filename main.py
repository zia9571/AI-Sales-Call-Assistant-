import time
from sentiment_analysis import analyze_sentiment, transcribe_with_chunks
from google_sheets import store_data_in_sheet
from env_setup import config

def main():
    # Transcription and sentiment analysis
    chunks = transcribe_with_chunks() 
    print("Final Chunks:")

    total_text = ""
    sentiment_scores = []

    # process the chunks to calculate the overall sentiment
    for i, (chunk, sentiment, score) in enumerate(chunks, 1):
        print(f"Chunk {i}: {chunk} | Sentiment: {sentiment} | Score: {score}")
        total_text += chunk + " "
        sentiment_scores.append(score if sentiment == "POSITIVE" else -score)

    # summarize the conversation and calculate overall sentiment
    overall_sentiment = "POSITIVE" if sum(sentiment_scores) > 0 else ("NEGATIVE" if sum(sentiment_scores) < 0 else "NEUTRAL")
    print("\nConversation Summary:")
    print(total_text.strip())
    print(f"Overall Sentiment: {overall_sentiment}")

    # store the results in Google Sheets
    store_data_in_sheet(config["google_sheet_id"], chunks, total_text.strip(), overall_sentiment)

if __name__ == "__main__":
    main()
