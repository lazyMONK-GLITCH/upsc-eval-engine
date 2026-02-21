import os
import sys
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Neo4jVector

# 1. Ignite Environment Variables
load_dotenv()

def ingest_document():
    print("🟢 PIPELINE ACTIVE: Initializing Industrial Document Loader...")
    
    loader = PyMuPDFLoader("rajasthan_history.pdf")
    docs = loader.load()
    print(f"📄 Extracted {len(docs)} pages from the document.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    print(f"✂️ Generated {len(chunks)} historical chunks.")

    if not chunks:
        print("❌ CRITICAL ERROR: 0 chunks generated. Scanned PDF detected.")
        sys.exit(1)

    print("🟢 PIPELINE ACTIVE: Connecting to Upgraded Gemini Embedding Core...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    print("🟢 PIPELINE ACTIVE: Fusing vectors into Neo4j Knowledge Graph...")
    
    # GOLD STANDARD 1: Initialize the Neo4j vector store with just the FIRST chunk.
    # This safely creates the database index without hitting rate limits.
    vector_store = Neo4jVector.from_documents(
        [chunks[0]],
        embeddings,
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
        index_name="vector",
        node_label="Chunk",
        text_node_property="text",
        embedding_node_property="embedding"
    )
    
    # GOLD STANDARD 2: Process the rest using safe micro-batches and .add_documents()
    batch_size = 15  # Ultra-safe batch size for Free Tier
    remaining_chunks = chunks[1:]  # Skip the first one we already did
    
    for i in range(0, len(remaining_chunks), batch_size):
        batch = remaining_chunks[i:i + batch_size]
        current_batch = (i // batch_size) + 1
        total_batches = (len(remaining_chunks) // batch_size) + 1
        
        print(f"🚀 Pushing Batch {current_batch} of {total_batches} ({len(batch)} chunks)...")
        
        # GOLD STANDARD 3: Fault-Tolerant Retry Loop
        max_retries = 3
        for attempt in range(max_retries):
            try:
                vector_store.add_documents(batch)
                break  # Success! Break out of the retry loop.
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    print(f"⚠️ API Limit hit on attempt {attempt + 1}. Sleeping for 60 seconds...")
                    time.sleep(60)
                else:
                    print(f"❌ Critical API Error: {e}")
                    raise e
        
        # Deliberate pacing to respect Google's strict 100 requests/minute free tier
        time.sleep(15)
            
    print("✅ INGESTION COMPLETE: Rajasthan History is fully integrated.")

if __name__ == "__main__":
    ingest_document()