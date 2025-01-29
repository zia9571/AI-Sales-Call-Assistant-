import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

class ProductRecommender:
    def __init__(self, product_data_path):
        self.data = pd.read_csv(product_data_path)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = self.model.encode(self.data['product_description'].tolist())
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def get_recommendations(self, query, top_n=5):
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(query_embedding, top_n)
        recommendations = []
        for i in indices[0]:
            recommendations.append(self.data.iloc[i]['product_title'] + ": " + self.data.iloc[i]['product_description'])
        return recommendations