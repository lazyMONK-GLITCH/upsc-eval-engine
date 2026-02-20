import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from database import get_db_connection

load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

print("🔌 Connecting to Neo4j...")
graph = get_db_connection()

print("🧠 Initializing Gemini Embeddings Model...")
embeddings_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

print("📄 Loading and parsing constitution.pdf...")
loader = PyPDFLoader("constitution.pdf")
docs = loader.load()

print(f"✂️ Chunking {len(docs)} pages of text...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
chunks = text_splitter.split_documents(docs)

print(f"⚙️ Vectorizing and pushing remaining chunks to Neo4j...")

doc_id = "indian_constitution_v1"
doc_title = "The Constitution of India"
topic_name = "Indian Polity and Constitution"
paper = "GS-2"

# RESUME CHECKPOINT: Skip the 800 chunks already in the database
START_CHUNK = 801

for i, chunk in enumerate(chunks):
    if i < START_CHUNK:
        continue
        
    chunk_id = f"{doc_id}_chunk_{i}"
    raw_text = chunk.page_content 
    
    # RATE LIMIT THROTTLE: 4-second delay to stay under 15 Requests Per Minute
    time.sleep(4)
    
    vector = embeddings_model.embed_query(raw_text)
    
    cypher_query = f"""
    MERGE (d:Document {{doc_id: "{doc_id}"}})
    ON CREATE SET d.title = "{doc_title}"
    
    MERGE (t:Topic {{name: "{topic_name}"}})
    ON CREATE SET t.paper = "{paper}"
    
    MERGE (c:Chunk {{chunk_id: "{chunk_id}"}})
    SET c.text = $text, 
        c.vector_embedding = $vector
        
    MERGE (c)-[:FROM_DOCUMENT]->(d)
    MERGE (c)-[:COVERS_TOPIC]->(t)
    """
    
    graph.query(cypher_query, params={"vector": vector, "text": raw_text})
    
    if i % 10 == 0:
        print(f"   ... Processed {i} chunks")

print(f"✅ Successfully ingested all chunks into the UPSC_BEAST Knowledge Graph.")