# Legal AI Assistant (RAG Chatbot)

An end-to-end full-stack AI chatbot that provides accurate legal information by leveraging a Retrieval-Augmented Generation (RAG) model. This platform uses a custom knowledge base to deliver context-aware responses and features a complete text-to-speech (TTS) and speech-to-text (STT) pipeline. 

## 🚀 Features

* **Retrieval-Augmented Generation (RAG):** The core of the system, enabling the chatbot to generate informed responses based on a provided set of legal documents.
* **Vector Database Integration:** Efficient semantic search and retrieval using a Pinecone vector database.
* **Speech-to-Text (STT):** Real-time voice input using the Web Speech API for a hands-free user experience.
* **Text-to-Speech (TTS):** Dynamic audio responses powered by the OpenAI API, providing an interactive conversational experience.
* **Full-Stack Architecture:** A complete web application built with a Flask backend and a modern JavaScript/HTML/CSS frontend.

## ⚙️ Technology Stack

**Backend:**
* **Python:** Primary programming language.
* **Flask:** Web framework for the RESTful API.
* **OpenAI API:** For LLM orchestration, embeddings, and TTS.
* **Pinecone:** Vector database for indexing and querying legal documents.

**Frontend:**
* **HTML, CSS, JavaScript:** For the user interface.
* **Web Speech API:** For speech-to-text functionality.
* **Font Awesome:** For icons.

## 📦 Setup and Installation

Follow these steps to get the project running on your local machine.

### **1. Clone the Repository**
Bash

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
(Replace your-username/your-repo-name with the actual path to your repository.)

2. Create and Activate a Virtual Environment
It's recommended to use a virtual environment to manage project dependencies.

Bash

# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS / Linux
python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
First, generate a requirements.txt file from your project's environment to ensure all necessary packages are included.

Bash

pip freeze > requirements.txt
Now, install the dependencies using the following command:

Bash

pip install -r requirements.txt
4. Configure Environment Variables
Create a file named .env in the root of your project and add your API keys.

Plaintext

OPENAI_API_KEY="your_openai_api_key_here"
PINECONE_API_KEY="your_pinecone_api_key_here"
PINECONE_ENVIRONMENT="your_pinecone_environment_here"
PINECONE_INDEX_NAME="your_pinecone_index_name_here"
(Replace the placeholder values with your actual keys and environment details.)

5. Run the Application
Start the Flask server from your terminal:

Bash

python main.py
The application will be accessible at http://127.0.0.1:5000.# Legal-AI-bot
