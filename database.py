import os
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph

load_dotenv()

def get_db_connection():
    return Neo4jGraph(
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD")
    )

def init_schema():
    graph = get_db_connection()
    print("🏗️ Upgrading Database Schema & Vector Indexes...")
    
    # 1. Purge the old 768-dimension index
    graph.query("DROP INDEX chunk_embeddings IF EXISTS")
    
    # 2. Enforce Digital Physics (Uniqueness Constraints)
    graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE")
    graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE")
    graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE")
    graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
    graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (p:PYQ) REQUIRE p.pyq_id IS UNIQUE")
    
    # 3. Create the new 3072-dimension Vector Index for gemini-embedding-001
    graph.query("""
    CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS 
    FOR (c:Chunk) ON (c.vector_embedding) 
    OPTIONS {indexConfig: {
      `vector.dimensions`: 3072,
      `vector.similarity_function`: 'cosine'
    }}
    """)
    print("✅ Schema Upgrade Complete.")

if __name__ == "__main__":
    init_schema()