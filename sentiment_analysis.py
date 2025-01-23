import os 
import json
import time
import speech_recognition as sr
from vosk import Model, KaldiRecognizer
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import login
from product_recommender import ProductRecommender
from objection_handler import load_objections, check_objections  # Ensure check_objections is imported
from objection_handler import ObjectionHandler
from env_setup import config
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the ProductRecommender
# Hugging Face API setup
huggingface_api_key = config["huggingface_api_key"]
login(token=huggingface_api_key)

# Sentiment Analysis Model
model_name = "tabularisai/multilingual-sentiment-analysis"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Vosk Speech Recognition Model
vosk_model_path = config["vosk_model_path"]

if not vosk_model_path:
    raise ValueError("Error: vosk_model_path is not set in the .env file.")

try:
    vosk_model = Model(vosk_model_path)
    print("Vosk model loaded successfully.")
except Exception as e:
    raise ValueError(f"Failed to load Vosk model: {e}")

recognizer = KaldiRecognizer(vosk_model, 16000)

# Initialize speech recognition
speech_recognizer = sr.Recognizer()
microphone = sr.Microphone()

# Function to analyze sentiment
def preprocess_text(text):
    """Preprocess text for better sentiment analysis."""
    return text.strip().lower()

def analyze_sentiment(text):
    """Analyze sentiment of the text using Hugging Face model."""
    try:
        if not text.strip():
            return "NEUTRAL", 0.0
        
        processed_text = preprocess_text(text)
        result = sentiment_analyzer(processed_text)[0]
        
        print(f"Sentiment Analysis Result: {result}")
        
        # Map raw labels to sentiments
        sentiment_map = {
            'Very Negative': "NEGATIVE",
            'Negative': "NEGATIVE",
            'Neutral': "NEUTRAL",
            'Positive': "POSITIVE",
            'Very Positive': "POSITIVE"
        }
        
        sentiment = sentiment_map.get(result['label'], "NEUTRAL")
        return sentiment, result['score']
        
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        return "NEUTRAL", 0.5

def transcribe_with_chunks(objections_dict):
    """Perform real-time transcription with sentiment analysis."""
    print("Say 'start listening' to begin transcription. Say 'stop listening' to stop.")
    is_listening = False
    chunks = []
    current_chunk = []
    chunk_start_time = time.time()

    # Initialize handlers with semantic search capabilities
    objection_handler = ObjectionHandler(r"C:\Users\shaik\Downloads\Sales Calls Transcriptions - Sheet3.csv")
    product_recommender = ProductRecommender(r"C:\Users\shaik\Downloads\Sales Calls Transcriptions - Sheet2.csv")

    # Load the embeddings model once
    model = SentenceTransformer('all-MiniLM-L6-v2')

    try:
        with microphone as source:
            print("Adjusting for ambient noise...")
            speech_recognizer.adjust_for_ambient_noise(source)
            while True:
                print("Listening...")

                # Listen for audio and capture it in real-time
                audio = speech_recognizer.listen(source, timeout=5)
                try:
                    text = speech_recognizer.recognize_google(audio)
                    print(f"Recognized Speech: {text}")

                    if "start listening" in text.lower():
                        is_listening = True
                        print("Listening started. Speak into the microphone.")
                        continue
                    elif "stop listening" in text.lower():
                        is_listening = False
                        print("Listening stopped.")
                        if current_chunk:
                            chunk_text = " ".join(current_chunk)
                            sentiment, score = analyze_sentiment(chunk_text)
                            chunks.append((chunk_text, sentiment, score))
                            current_chunk = []
                        continue

                    if is_listening and text.strip():
                        print(f"Transcription: {text}")
                        current_chunk.append(text)

                        if time.time() - chunk_start_time > 3:
                            if current_chunk:
                                chunk_text = " ".join(current_chunk)
                                
                                # Always process sentiment
                                sentiment, score = analyze_sentiment(chunk_text)
                                chunks.append((chunk_text, sentiment, score))

                                # Get objection responses and check similarity score
                                query_embedding = model.encode([chunk_text])
                                distances, indices = objection_handler.index.search(query_embedding, 1)
                                
                                # If similarity is high enough, show objection response
                                if distances[0][0] < 1.5:  # Threshold for similarity
                                    responses = objection_handler.handle_objection(chunk_text)
                                    if responses:
                                        print("\nSuggested Response:")
                                        for response in responses:
                                            print(f"→ {response}")
                                
                                # Get product recommendations and check similarity score
                                distances, indices = product_recommender.index.search(query_embedding, 1)
                                
                                # If similarity is high enough, show recommendations
                                if distances[0][0] < 1.5:  # Threshold for similarity
                                    recommendations = product_recommender.get_recommendations(chunk_text)
                                    if recommendations:
                                        print(f"\nRecommendations for this response:")
                                        for idx, rec in enumerate(recommendations, 1):
                                            print(f"{idx}. {rec}")
                                
                                print("\n")
                                current_chunk = []
                                chunk_start_time = time.time()

                except sr.UnknownValueError:
                    print("Sorry, I couldn't understand that.")
                except sr.RequestError:
                    print("Could not request results from Google Speech Recognition service.")

    except KeyboardInterrupt:
        print("\nExiting...")

    return chunks

if __name__ == "__main__":
    objections_file_path = r"C:\Users\shaik\Downloads\Sales Calls Transcriptions - Sheet3.csv"
    objections_dict = load_objections(objections_file_path)
    transcribed_chunks = transcribe_with_chunks(objections_dict)
    print("Final transcriptions and sentiments:", transcribed_chunks)
