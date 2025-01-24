from dotenv import load_dotenv
import os
from vosk import Model, KaldiRecognizer
import pyaudio
import json

load_dotenv()

vosk_model_path = os.getenv("vosk_model_path")

if not vosk_model_path:
    print("Error: vosk_model_path is not set in the .env file.")
    exit()

try:
    model = Model(vosk_model_path)
    print("Vosk model loaded successfully.")
except Exception as e:
    print(f"Failed to load Vosk model: {e}")
    exit()

recognizer = KaldiRecognizer(model, 16000)
audio = pyaudio.PyAudio()

stream = audio.open(format=pyaudio.paInt16, 
                    channels=1, 
                    rate=16000, 
                    input=True, 
                    frames_per_buffer=4000)
stream.start_stream()

print("Say 'start listening' to begin transcription and 'stop listening' to stop.")

is_listening = False

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
                print("Listening stopped. Say 'start listening' to resume.")
                continue

            if is_listening:
                print(f"Transcription: {text}")
        else:
            chunk_result = recognizer.PartialResult()
            chunk_text = json.loads(chunk_result)["partial"]

            if is_listening and chunk_text:
                print(f"chunk: {chunk_text}", end="\r")

except KeyboardInterrupt:
    print("\nExiting...")
    stream.stop_stream()
    stream.close()
    audio.terminate()
