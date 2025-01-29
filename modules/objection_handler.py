import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

def load_objections(file_path):
    """Load objections from a CSV file into a dictionary."""
    try:
        objections_df = pd.read_csv(file_path)
        objections_dict = {}
        for index, row in objections_df.iterrows():
            objections_dict[row['Customer Objection']] = row['Salesperson Response']
        return objections_dict
    except Exception as e:
        print(f"Error loading objections: {e}")
        return {}

def check_objections(text, objections_dict):
    """Check for objections in the given text and return responses."""
    responses = []
    for objection, response in objections_dict.items():
        if objection.lower() in text.lower():
            responses.append(response)
    return responses

class ObjectionHandler:
    def __init__(self, objection_data_path):
        self.data = pd.read_csv(objection_data_path)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = self.model.encode(self.data['Customer Objection'].tolist())
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def handle_objection(self, query, top_n=1):
        """Handle objections using embeddings."""
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(query_embedding, top_n)
        responses = []
        for i in indices[0]:
            responses.append(self.data.iloc[i]['Salesperson Response'])
        return responses