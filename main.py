from sentiment_analysis import transcribe_with_chunks
from google_sheets import store_data_in_sheet
from env_setup import config
from product_recommender import get_recommendations  

def main():
    chunks = transcribe_with_chunks()  
    total_text = ""
    sentiment_scores = []

    for chunk, sentiment, score in chunks:
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
                "sunglasses", "scarf", "handbag", "wallet", "belt", "earrings", "hat",
                "jacket", "blazer", "coat", "raincoat", "sweater",
                "cotton", "denim", "wool", "leather", "polyester", "satin", "velvet",
                "white", "black", "red", "blue", "green", "yellow", "pink", "beige", "gray", "brown", "purple",
                "socks", "luggage", "bag"  # Updated keyword list
            ]):
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

    overall_sentiment = "POSITIVE" if sum(sentiment_scores) > 0 else (
        "NEGATIVE" if sum(sentiment_scores) < 0 else "NEUTRAL"
    )

    print("\nConversation Summary:")
    print(total_text.strip())
    print(f"Overall Sentiment: {overall_sentiment}")

    store_data_in_sheet(config["google_sheet_id"], chunks, total_text.strip(), overall_sentiment)

if __name__ == "__main__":
    main()
