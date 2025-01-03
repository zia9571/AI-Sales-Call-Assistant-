# Import necessary modules (ensure this is at the top of your file)
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

# Load data and create FAISS index (use your existing code for this)
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

# Function to fetch top N similar products
def get_recommendations(query, top_n=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_n)
    recommendations = []
    for i in indices[0]:
        recommendations.append(data.iloc[i]['product_title'] + ": " + data.iloc[i]['product_description'])
    return recommendations

# Interactive query testing
while True:
    user_query = input("\nEnter a query (or type 'exit' to quit): ")
    if user_query.lower() == 'exit':
        print("Exiting the recommendation system.")
        break
    recommendations = get_recommendations(user_query)
    print(f"\nTop recommendations for '{user_query}':")
    for idx, rec in enumerate(recommendations, 1):
        print(f"{idx}. {rec}")
