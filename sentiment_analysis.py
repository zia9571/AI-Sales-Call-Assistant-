from dotenv import load_dotenv
import os
import time
import json
from vosk import Model, KaldiRecognizer
import pyaudio
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import login
from env_setup import config

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

def transcribe_with_chunks():
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

                    if time.time() - chunk_start_time > 3:
                        if current_chunk:
                            chunk_text = " ".join(current_chunk)
                            sentiment, score = analyze_sentiment(chunk_text)
                            chunks.append((chunk_text, sentiment, score))
                            print(f"Chunk: {chunk_text} | Sentiment: {sentiment} | Score: {score}")
                            current_chunk = []
                        chunk_start_time = time.time()

    except KeyboardInterrupt:
        print("\nExiting...")
        stream.stop_stream()
        stream.close()
        audio.terminate()
        return chunks
