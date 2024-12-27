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
vosk_model_path = os.getenv("vosk_model_path")

if not vosk_model_path:
    print("Error: vosk_model_path is not set in the .env file.")
    exit()

try:
    vosk_model = Model(vosk_model_path)
    print("Vosk model loaded successfully.")
except Exception as e:
    print(f"Failed to load Vosk model: {e}")
    exit()

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

print("Say 'start listening' to begin transcription and sentiment analysis. Say 'stop listening' to stop.")

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

            # Command to start/stop listening
            if "start listening" in text.lower():
                is_listening = True
                print("Listening started. Speak into the microphone.")
                continue
            elif "stop listening" in text.lower():
                is_listening = False
                print("Listening stopped. Say 'start listening' to resume.")
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    sentiment, score = analyze_sentiment(chunk_text)
                    chunks.append((chunk_text, sentiment, score))
                    current_chunk = []
                continue

            # Process transcription if actively listening
            if is_listening:
                print(f"Transcription: {text}")
                current_chunk.append(text)

                # Check for pauses to finalize a chunk
                if time.time() - chunk_start_time > 3:
                    if current_chunk:
                        chunk_text = " ".join(current_chunk)
                        sentiment, score = analyze_sentiment(chunk_text)
                        chunks.append((chunk_text, sentiment, score))
                        print(f"Chunk saved: {chunk_text} | Sentiment: {sentiment} | Score: {score}")
                        current_chunk = []

                chunk_start_time = time.time()

except KeyboardInterrupt:
    print("\nExiting...")
    stream.stop_stream()
    stream.close()
    audio.terminate()

# Print the final results
print("\nFinal Chunks:")

total_text = ""
sentiment_scores = []

for i, (chunk, sentiment, score) in enumerate(chunks, 1):
    print(f"Chunk {i}: {chunk} | Sentiment: {sentiment} | Score: {score}")
    total_text += chunk + " "
    sentiment_scores.append(score if sentiment == "POSITIVE" else -score)

overall_sentiment = "POSITIVE" if sum(sentiment_scores) > 0 else ("NEGATIVE" if sum(sentiment_scores) < 0 else "NEUTRAL")
print("\nConversation Summary:")
print(total_text.strip())
print(f"Overall Sentiment: {overall_sentiment}")
