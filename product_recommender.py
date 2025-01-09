# Import necessary modules
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

# Load data and create FAISS index
file_path = "C:\\Users\\shaik\\Downloads\\Sales Calls Transcriptions - Sheet2.csv"  # Adjust to your actual CSV location
data = pd.read_csv(file_path)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
product_descriptions = data['product_description'].tolist()
embeddings = model.encode(product_descriptions)

# Create and populate FAISS index
dimension = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Expanded Keyword List
keywords = [
    "dress", "floral dress", "maxi dress", "velvet dress", "sundress",
    "shirt", "casual shirt", "formal shirt", "t-shirt", "pajama set", "blouse",
    "trousers", "jeans", "chinos", "joggers", "sweatpants",
    "skirt", "denim skirt", "maxi skirt", "pencil skirt",
    "shoes", "running shoes", "high heels", "loafers", "sandals", "boots", "sneakers",
    "sunglasses", "scarf", "handbag", "wallet", "belt", "earrings", "hat",
    "jacket", "blazer", "coat", "raincoat", "sweater",
    "cotton", "denim", "wool", "leather", "polyester", "satin", "velvet",
    "white", "black", "red", "blue", "green", "yellow", "pink", "beige", "gray", "brown", "purple",
    "socks", "luggage", "bag"  # Added keywords for socks and luggage
]

# Function to fetch top N similar products
def get_recommendations(query, top_n=5):
    # Check for relevant keywords in the query
    if any(keyword in query.lower() for keyword in keywords):
        query_embedding = model.encode([query])
        distances, indices = index.search(query_embedding, top_n)
        recommendations = []
        for i in indices[0]:
            recommendations.append(data.iloc[i]['product_title'] + ": " + data.iloc[i]['product_description'])
        return recommendations
    else:
        print(f"No relevant keywords found in the query: '{query}'. Skipping recommendations.")
        return []

# Interactive query testing
if __name__ == "__main__":
    while True:
        user_query = input("\nEnter a query (or type 'exit' to quit): ")
        if user_query.lower() == 'exit':
            print("Exiting the recommendation system.")
            break
        recommendations = get_recommendations(user_query)
        if recommendations:
            print(f"\nTop recommendations for '{user_query}':")
            for idx, rec in enumerate(recommendations, 1):
                print(f"{idx}. {rec}")
