import os
import time
import speech_recognition as sr
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import login
from env_setup import config

# Log in 
huggingface_api_key = config["huggingface_api_key"]
login(token=huggingface_api_key)

#multilingual sentiment analysis model
model_name = "tabularisai/multilingual-sentiment-analysis"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# recognizer
r = sr.Recognizer()

def analyze_sentiment(text):
    """Analyze sentiment of the text using Hugging Face model."""
    result = sentiment_analyzer(text)[0]
    sentiment = result['label']
    score = result['score']
    
    # sentiment mapping
    sentiment_map = {
        "LABEL_0": "VERY NEGATIVE",  
        "LABEL_1": "NEGATIVE",
        "LABEL_2": "NEUTRAL",
        "LABEL_3": "POSITIVE",
        "LABEL_4": "VERY POSITIVE"
    }
    
    #  sentiment label and score
    sentiment = sentiment_map.get(result['label'], "NEUTRAL")
    
    return sentiment, score

def transcribe_with_chunks():
    """
    Transcribe audio input in real-time and create chunks based on pauses.
    Returns a list of chunks with their sentiment analysis.
    """
    chunks = []
    current_chunk = []
    chunk_start_time = time.time()

    print("Say 'start recording' to begin and 'stop recording' to stop.")
    is_active = False

    while True:
        try:
            with sr.Microphone() as source:
                # ambient noise
                r.adjust_for_ambient_noise(source, duration=0.2)
                
                if not is_active:
                    print("Waiting for 'start recording'...")
                    audio = r.listen(source, timeout=None)
                    command = r.recognize_google(audio).lower()

                    if "start recording" in command:
                        is_active = True
                        print("Recording started")
                    continue

                audio = r.listen(source, timeout=None, phrase_time_limit=5)
                text = r.recognize_google(audio).lower()

                if "stop recording" in text:
                    print("Recording stopped")
                    if current_chunk:
                        chunk_text = " ".join(current_chunk)
                        sentiment, score = analyze_sentiment(chunk_text)
                        chunks.append((chunk_text, sentiment, score))
                    break

                current_chunk.append(text)
                print(f"Speaker: {text}")

                # pause (3 seconds) to finalize a chunk
                if time.time() - chunk_start_time > 3:
                    if current_chunk:
                        chunk_text = " ".join(current_chunk)
                        sentiment, score = analyze_sentiment(chunk_text)
                        chunks.append((chunk_text, sentiment, score))
                        print(f"Chunk saved: {chunk_text} | Sentiment: {sentiment} | Score: {score}")
                        current_chunk = []

                chunk_start_time = time.time()

        except sr.UnknownValueError:
            continue
        except sr.RequestError as e:
            print(f"Could not request results; {e}")

    return chunks

if __name__ == "__main__":
    chunks = transcribe_with_chunks()
    print("Final Chunks:")

    total_text = ""
    sentiment_scores = []

    for i, (chunk, sentiment, score) in enumerate(chunks, 1):
        print(f"Chunk {i}: {chunk} | Sentiment: {sentiment} | Score: {score}")
        total_text += chunk + " "
        sentiment_scores.append(score if sentiment == "POSITIVE" else -score)

    # summarize and calculate overall sentiment
    overall_sentiment = "POSITIVE" if sum(sentiment_scores) > 0 else ("NEGATIVE" if sum(sentiment_scores) < 0 else "NEUTRAL")
    print("\nConversation Summary:")
    print(total_text.strip())
    print(f"Overall Sentiment: {overall_sentiment}")
