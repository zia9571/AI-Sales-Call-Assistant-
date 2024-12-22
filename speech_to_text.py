import time
import speech_recognition as sr
import pyttsx3

# Initialize recognizer
r = sr.Recognizer()

def transcribe_with_chunks():
    """
    Transcribe audio input in real-time and create chunks based on pauses.
    Returns a list of chunks (each chunk is a string of transcribed text).
    """
    chunks = []
    current_chunk = []
    chunk_start_time = time.time()

    print("Say 'start recording' to begin and 'stop recording' to stop.")
    is_active = False

    while True:
        try:
            with sr.Microphone() as source:
                # Adjust for ambient noise
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
                        chunks.append(" ".join(current_chunk))
                    break

                current_chunk.append(text)
                print(f"Speaker: {text}")

                # Detect pause (3 seconds) to finalize a chunk
                if time.time() - chunk_start_time > 3:
                    if current_chunk:
                        chunks.append(" ".join(current_chunk))
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
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}: {chunk}")
