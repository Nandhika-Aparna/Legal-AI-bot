import os
import json
import base64
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from dotenv import load_dotenv
from flask_cors import CORS

# Load environment variables
load_dotenv()

# Initialize API clients and server
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("INDEX_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not all([PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME, OPENAI_API_KEY]):
    raise ValueError("Missing one or more required environment variables.")

try:
    # Use the new Pinecone client syntax (recommended)
    pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    index = pinecone_client.Index(os.getenv("INDEX_NAME"))
    print("Pinecone client initialized successfully.")
    
    # Check if the index is ready
    if not pinecone_client.describe_index(os.getenv("INDEX_NAME")).status['ready']:
        print("Warning: Pinecone index is not ready. Please check your dashboard.")

except Exception as e:
    # Handle the error gracefully if the connection fails
    print(f"Error initializing Pinecone: {e}")
    # You might want to exit the application or handle this gracefully.
    index = None  # Ensure 'index' is defined even on failure

openai_client = OpenAI(api_key=OPENAI_API_KEY)
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# --- Chat History Management Functions ---
def get_today_chat_file() -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    return f"chat_history/{today}.json"

def load_chat_history() -> list:
    filename = get_today_chat_file()
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_chat_history(history: list):
    filename = get_today_chat_file()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

# --- Existing RAG Function ---
def get_conversational_response(query, context):
    messages = [
        {
            "role": "system",
            "content": "You are a legal research chatbot. Based on the provided legal documents and your general knowledge, answer the user's question accurately and professionally. Cite specific document excerpts where applicable. If the provided context is not sufficient, state that you cannot provide a detailed answer based on the given documents. Do not hallucinate or create legal information."
        },
        {
            "role": "user",
            "content": f"Based on the following legal documents, answer this question: {query}\n\nDocuments:\n{context}"
        }
    ]
    
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    
    return response.choices[0].message.content

# --- NEW: Text-to-Speech Function ---
def get_tts_audio(text: str) -> str:
    """Converts text to speech and returns the Base64-encoded audio data."""
    audio_response = openai_client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text
    )
    # The audio_response is a streaming object. Read it into a buffer.
    audio_buffer = audio_response.read()
    # Encode the audio buffer to a Base64 string
    base64_audio = base64.b64encode(audio_buffer).decode('utf-8')
    return base64_audio

# --- API Endpoints (Updated) ---


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_query = data.get("query")

    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    try:
        # RAG pipeline steps (same as before)
        query_embedding_response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=[user_query]
        )
        query_embedding = query_embedding_response.data[0].embedding
        search_results = index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True
        )
        context_chunks = [match.metadata["text"] for match in search_results.matches]
        context_string = "\n\n".join(context_chunks)
        ai_response_text = get_conversational_response(user_query, context_string)

        # --- NEW: Get TTS Audio for the AI's response ---
        ai_response_audio = get_tts_audio(ai_response_text)
        
        # Add a print statement to verify the audio data is not empty
        if ai_response_audio:
            print("Successfully generated audio data.")
        else:
            print("Error: Audio data is empty.")

        # History saving logic
        chat_history = load_chat_history()
        chat_history.append({"role": "user", "content": user_query})
        chat_history.append({"role": "assistant", "content": ai_response_text})
        chat_history.append({"role": "assistant", "content": ai_response_audio})
        save_chat_history(chat_history)
        
        # Return both the text and the audio
        return jsonify({
            "chatHistory": chat_history,
            "audioResponse": ai_response_audio
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/get_history", methods=["GET"])
def get_history():
    chat_history = load_chat_history()
    return jsonify({"chatHistory": chat_history})

if __name__ == "__main__":
    os.makedirs("chat_history", exist_ok=True)
    app.run(debug=True, port=5000)