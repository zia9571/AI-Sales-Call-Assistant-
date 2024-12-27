from dotenv import load_dotenv
import os
from vosk import Model, KaldiRecognizer
import pyaudio
import json

# Load environment variables from .env file
load_dotenv()

# Get the Vosk model path from the environment variable
vosk_model_path = os.getenv("vosk_model_path")

if not vosk_model_path:
    print("Error: vosk_model_path is not set in the .env file.")
    exit()

# Initialize the Vosk model
try:
    model = Model(vosk_model_path)
    print("Vosk model loaded successfully.")
except Exception as e:
    print(f"Failed to load Vosk model: {e}")
    exit()

# Initialize recognizer and audio input
recognizer = KaldiRecognizer(model, 16000)
audio = pyaudio.PyAudio()

# Open audio stream
stream = audio.open(format=pyaudio.paInt16, 
                    channels=1, 
                    rate=16000, 
                    input=True, 
                    frames_per_buffer=4000)
stream.start_stream()

print("Say 'start listening' to begin transcription and 'stop listening' to stop.")

# State management
is_listening = False

try:
    while True:
        data = stream.read(4000, exception_on_overflow=False)

        if recognizer.AcceptWaveform(data):
            result = recognizer.Result()
            text = json.loads(result)["text"]
            
            # Check for commands to start or stop listening
            if "start listening" in text.lower():
                is_listening = True
                print("Listening started. Speak into the microphone.")
                continue
            elif "stop listening" in text.lower():
                is_listening = False
                print("Listening stopped. Say 'start listening' to resume.")
                continue

            # Transcribe if actively listening
            if is_listening:
                print(f"Transcription: {text}")
        else:
            # Handle partial results if needed
            chunk_result = recognizer.PartialResult()
            chunk_text = json.loads(chunk_result)["partial"]

            # Display partial transcription only if actively listening
            if is_listening and chunk_text:
                print(f"chunk: {chunk_text}", end="\r")

except KeyboardInterrupt:
    print("\nExiting...")
    stream.stop_stream()
    stream.close()
    audio.terminate()
