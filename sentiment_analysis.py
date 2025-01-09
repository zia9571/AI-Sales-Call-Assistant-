from dotenv import load_dotenv
import os
import time
import json
from vosk import Model, KaldiRecognizer
import pyaudio
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import login
from env_setup import config
from product_recommender import get_recommendations  # Import the recommendation function
from objection_handler import load_objections, check_objections

# Load environment variables
load_dotenv()

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
audio = pyaudio.PyAudio()

stream = audio.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    frames_per_buffer=4000)
stream.start_stream()

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
    "white", "black", "red", "blue", "green", "yellow", "pink", "beige", "gray", "brown", "purple"
]

# Function to analyze sentiment
def analyze_sentiment(text):
    """Analyze sentiment of the text using Hugging Face model."""
    result = sentiment_analyzer(text)[0]
    sentiment_map = {
        "LABEL_0": "VERY NEGATIVE",
        "LABEL_1": "NEGATIVE",
        "LABEL_2": "NEUTRAL",
        "LABEL_3": "POSITIVE",
        "LABEL_4": "VERY POSITIVE"
    }
    sentiment = sentiment_map.get(result['label'], "NEUTRAL")
    return sentiment, result['score']

objections_file_path = r"C:\Users\shaik\Downloads\Sales Calls Transcriptions - Sheet3.csv"
objections_dict = load_objections(objections_file_path)

# Existing imports and code...

# Function to analyze sentiment
def analyze_sentiment(text):
    """Analyze sentiment of the text using Hugging Face model."""
    result = sentiment_analyzer(text)[0]
    sentiment_map = {
        "LABEL_0": "VERY NEGATIVE",
        "LABEL_1": "NEGATIVE",
        "LABEL_2": "NEUTRAL",
        "LABEL_3": "POSITIVE",
        "LABEL_4": "VERY POSITIVE"
    }
    sentiment = sentiment_map.get(result['label'], "NEUTRAL")
    return sentiment, result['score']

# Load objections
objections_file_path = r"C:\Users\shaik\Downloads\Sales Calls Transcriptions - Sheet3.csv"
objections_dict = load_objections(objections_file_path)

def transcribe_with_chunks(objections_dict):
    """Perform real-time transcription with sentiment analysis."""
    print("Say 'start listening' to begin transcription. Say 'stop listening' to stop.")
    is_listening = False
    chunks = []
    current_chunk = []
    chunk_start_time = time.time()

    try:
        while True:
            data = stream.read(4000, exception_on_overflow=False)

            if recognizer.AcceptWaveform(data):
                result = recognizer.Result()
                text = json.loads(result)["text"]

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

                if is_listening:
                    print(f"Transcription: {text}")
                    current_chunk.append(text)

                    # Check for objections in the current chunk
                    objection_responses = check_objections(text, objections_dict)
                    for objection, response in objection_responses:
                        print(f"Objection detected: '{objection}'")
                        print(f"Suggested response: {response}")

                    if time.time() - chunk_start_time > 3:
                        if current_chunk:
                            chunk_text = " ".join(current_chunk)
                            sentiment, score = analyze_sentiment(chunk_text)
                            chunks.append((chunk_text, sentiment, score))
                            print(f"Chunk: {chunk_text} | Sentiment: {sentiment} | Score: {score}")

                            # Check for relevant keywords and get recommendations
                            if any(keyword in chunk_text.lower() for keyword in keywords):
                                recommendations = get_recommendations(chunk_text)
                                if recommendations:
                                    print(f"Recommendations for chunk: '{chunk_text}'")
                                    for idx, rec in enumerate(recommendations, 1):
                                        print(f"{idx}. {rec}")
                                else:
                                    print("No recommendations found.")
                            else:
                                print(f"No relevant keywords found in the chunk: '{chunk_text}'. Skipping recommendations.")

                            current_chunk = []
                        chunk_start_time = time.time()

    except KeyboardInterrupt:
        print("\nExiting...")
        stream.stop_stream()

    # Ensure to return the chunks at the end of the function
    return chunks

# Add this block to allow running the script directly
if __name__ == "__main__":
    transcribed_chunks = transcribe_with_chunks(objections_dict)
    print("Final transcriptions and sentiments:", transcribed_chunks)
