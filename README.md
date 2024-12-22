# AI-Sales-Call-Assistant-
Overview
This project involves creating an AI-driven assistant that analyzes real-time speech during sales calls. The system processes audio input from the microphone, transcribes it to text, performs sentiment analysis on each conversation chunk, and provides a summary of the entire call. The data is then stored in a Google Sheet, enabling easy review and analysis of the conversation's tone and content.

Features
Real-Time Speech-to-Text: Converts live speech during sales calls into text.
Sentiment Analysis: Detects the sentiment (Positive, Negative, or Neutral) for each conversation chunk using a Hugging Face model.
Conversation Chunking: The speech is segmented into chunks based on pauses (3 seconds or more).
Conversation Summary: Summarizes the entire conversation after the call ends.
Google Sheets Integration: Stores conversation chunks, sentiment analysis results, and a summary in a Google Sheet for easy tracking and review.

Requirements
To set up and run this project, you will need the following:

Python 3.7 or higher
Google Cloud API (for Google Sheets integration)
Hugging Face account (for accessing the sentiment analysis model)
Required Python Libraries:
google-api-python-client: For Google Sheets API interaction.
google-auth-httplib2: Authentication for Google Sheets API.
google-auth-oauthlib: OAuth support for Google Sheets API.
huggingface-hub: Hugging Face API interaction.
transformers: For sentiment analysis using pre-trained models.
torch: Required for the Hugging Face models.
speechrecognition: For real-time speech recognition.
pyaudio: For microphone input.
pandas: Optional, for data handling (if needed).
pyttex3: For LaTeX rendering (if applicable).
Installation
Clone this repository to your local machine:

git clone https://github.com/yourusername/real-time-ai-sales-call-assistant.git
Navigate to the project directory:

cd real-time-ai-sales-call-assistant
Install the required Python libraries:

pip install -r requirements.txt

Set up Google Sheets API credentials and Hugging Face API credentials. Follow the instructions in the respective documentation to obtain your API keys.

Configuration
Google Sheets API Configuration
Create a project in Google Cloud Console.
Enable the Google Sheets API for your project.
Download the OAuth 2.0 credentials JSON file and save it as credentials.json in your project directory.
Share the target Google Sheet with the email ID mentioned in your credentials.json.
Hugging Face API Configuration
Create an account on Hugging Face.

Run the script to start analyzing sales calls:

python sales_call_assistant.py
During the call, say “start recording” to begin, and “stop recording” to end the session.

The conversation will be processed, and chunks will be analyzed for sentiment. After the call ends, the summary and sentiment analysis will be stored in your Google Sheet.

Example Output
Chunks:

Chunk 1: “Hello, how can I help you today?” | Sentiment: Positive | Score: 0.98
Chunk 2: “I have some concerns about the price.” | Sentiment: Negative | Score: 0.12
Chunk 3: “Let me explain the benefits to you.” | Sentiment: Neutral | Score: 0.50
Conversation Summary: "Hello, how can I help you today? I have some concerns about the price. Let me explain the benefits to you."

Overall Sentiment: Neutral

Contributing
Feel free to fork this repository and make changes. If you have suggestions or improvements, please create a pull request. Contributions are welcome!

License
This project is licensed under the MIT License - see the LICENSE file for details.

