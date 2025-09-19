import os
import tqdm
import asyncio
import uuid
import glob
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import AsyncOpenAI
from openai import APIConnectionError

# Load environment variables
load_dotenv()

# --- Configuration ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")
DATA_DIR = "data"
EMBEDDING_MODEL_ID = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
OPENAI_BATCH_SIZE = 500
PINECONE_BATCH_SIZE = 50
MAX_CONCURRENT_REQUESTS = 10 # <-- A safe number to prevent memory errors

# --- Main Ingestion Logic ---
async def create_embeddings_and_ingest():
    print("Initializing Pinecone client...")
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Check for and create the index
    if INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating Pinecone index: {INDEX_NAME}...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region=PINECONE_ENVIRONMENT)
        )
    index = pc.Index(INDEX_NAME)

    print("Loading documents from data directory...")
    documents = []
    pdf_files = glob.glob(os.path.join(DATA_DIR, '**/*.pdf'), recursive=True)
    
    # Use tqdm to show a progress bar for each file
    for file_path in tqdm.tqdm(pdf_files, desc="Loading documents"):
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())

    print(f"Loaded {len(documents)} total documents.")
    
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    print(f"Split documents into {len(docs)} chunks.")

    print(f"Generating embeddings for {len(docs)} chunks and uploading to Pinecone...")
    
    # Increase the timeout for the OpenAI client
    openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY, timeout=60.0)
    
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async def embed_and_upload_batch(chunk_batch):
        async with semaphore:
            texts = [doc.page_content for doc in chunk_batch]
            
            retries = 3
            delay = 2  # seconds
            
            for attempt in range(retries):
                try:
                    response = await openai_client.embeddings.create(
                        model=EMBEDDING_MODEL_ID,
                        input=texts
                    )
                    
                    # If the request succeeds, break the retry loop
                    break
                except APIConnectionError as e:
                    if attempt < retries - 1:
                        print(f"Connection error on attempt {attempt + 1}/{retries}. Retrying in {delay}s...")
                        await asyncio.sleep(delay)
                        delay *= 2  # Exponential backoff
                    else:
                        # If all retries fail, re-raise the exception
                        print(f"Failed to connect after {retries} attempts. Skipping this batch.")
                        return # Exit the function, skipping the upsert
            else:
                # This block runs if the for loop completes without breaking (i.e., all retries failed)
                return

            # If we get here, the request was successful
            vectors_to_upsert = []
            for item, doc in zip(response.data, chunk_batch):
                vectors_to_upsert.append({
                            "id": str(uuid.uuid4()),
                            "values": item.embedding,
                            "metadata": {
                                "source": doc.metadata.get('source', 'unknown'),
                                "text": doc.page_content # <-- Add this line
                            }
                        })
            
            for i in range(0, len(vectors_to_upsert), PINECONE_BATCH_SIZE):
                batch = vectors_to_upsert[i:i + PINECONE_BATCH_SIZE]
                try:
                    index.upsert(vectors=batch)
                except Exception as e:
                    print(f"Error during upsert for a sub-batch: {e}")
                    
    tasks = [
        embed_and_upload_batch(docs[i:i + OPENAI_BATCH_SIZE])
        for i in range(0, len(docs), OPENAI_BATCH_SIZE)
    ]
    
    # Run all tasks with a progress bar
    await tqdm.asyncio.tqdm.gather(*tasks, desc="Embedding & Uploading")
    
    print("\nIngestion complete.")

if __name__ == "__main__":
    asyncio.run(create_embeddings_and_ingest())