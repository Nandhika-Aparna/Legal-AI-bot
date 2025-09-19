import os
import json
from datetime import datetime
from flask import Flask, request, jsonify
from pinecone import Pinecone
from openai import OpenAI
from dotenv import load_dotenv
from flask_cors import CORS

# Load environment variables from .env file
load_dotenv()

# Initialize API clients and server
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("INDEX_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Check for required environment variables
if not all([PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME, OPENAI_API_KEY]):
    raise ValueError("Missing one or more required environment variables.")

# Initialize Pinecone and OpenAI clients
pinecone_client = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
openai_client = OpenAI(api_key=OPENAI_API_KEY)
index = pinecone_client.Index(PINECONE_INDEX_NAME)

app = Flask(__name__)
CORS(app)

# --- Chat History Management Functions (New) ---
def get_today_chat_file() -> str:
    """Returns the filename for today's chat history."""
    today = datetime.now().strftime("%Y-%m-%d")
    return f"chat_history/{today}.json"

def load_chat_history() -> list:
    """Loads chat history from today's file or returns an empty list if it doesn't exist."""
    filename = get_today_chat_file()
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_chat_history(history: list):
    """Saves the chat history to today's file."""
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

# --- API Endpoints (Updated) ---
@app.route("/log_error", methods=["POST"])
def log_error():
    """
    Receives an error message from the frontend and logs it to the server's console.
    """
    try:
        error_data = request.json
        error_message = error_data.get("message", "Unknown frontend error")
        # Print the error to the bash terminal
        print(f"\n--- Frontend Error Log ---")
        print(f"Error Message: {error_message}")
        print("--------------------------\n")
        return jsonify({"status": "success"}), 200
    except Exception as e:
        print(f"Failed to log frontend error: {e}")
        return jsonify({"status": "failure"}), 500

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_query = data.get("query")

    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    try:
        # Step 1: Create embedding for the user's query
        query_embedding_response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=[user_query]
        )
        query_embedding = query_embedding_response.data[0].embedding

        # Step 2: Search Pinecone for relevant document chunks
        search_results = index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True
        )

        # Step 3: Build context from search results
        context_chunks = [match.metadata["text"] for match in search_results.matches]
        context_string = "\n\n".join(context_chunks)

        # Step 4: Get a conversational response from GPT-4o
        ai_response = get_conversational_response(user_query, context_string)
        
        # --- History Saving Logic (New) ---
        # Load existing chat history
        chat_history = load_chat_history()
        
        # Append the new user-assistant exchange to the history
        chat_history.append({"role": "user", "content": user_query})
        chat_history.append({"role": "assistant", "content": ai_response})
        
        # Save the updated history
        save_chat_history(chat_history)
        
        # Return the entire history, so the frontend can display it
        return jsonify({"chatHistory": chat_history})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/get_history", methods=["GET"])
def get_history():
    """
    New endpoint to retrieve the full chat history for the day.
    """
    chat_history = load_chat_history()
    return jsonify({"chatHistory": chat_history})

if __name__ == "__main__":
    # Create the chat_history directory if it doesn't exist
    os.makedirs("chat_history", exist_ok=True)
    app.run(debug=True, port=5000)