import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from database import get_db_connection

load_dotenv()

# Bridge LangChain and Gemini keys
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

def retrieve_context(query: str, top_k: int = 5) -> str:
    print(f"🔎 Activating Neo4j Vector Search for: '{query}'")
    
    graph = get_db_connection()
    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    # 1. Convert the user's text into a 3072-dimension math vector
    query_vector = embeddings_model.embed_query(query)
    
    # 2. Execute strict Cypher Vector Search against the graph
    cypher_query = """
    CALL db.index.vector.queryNodes('chunk_embeddings', $top_k, $query_vector)
    YIELD node, score
    MATCH (node)-[:FROM_DOCUMENT]->(d:Document)
    MATCH (node)-[:COVERS_TOPIC]->(t:Topic)
    RETURN node.text AS text, score, d.title AS document, t.name AS topic
    """
    
    results = graph.query(cypher_query, params={"query_vector": query_vector, "top_k": top_k})
    
    # 3. Format the retrieved nodes into a structured context string
    if not results:
        return "No relevant constitutional context found in the database."
        
    context_blocks = []
    for res in results:
        context_blocks.append(
            f"[Source: {res['document']} | Topic: {res['topic']} | Relevance: {res['score']:.2f}]\n{res['text']}"
        )
        
    return "\n\n---\n\n".join(context_blocks)

if __name__ == "__main__":
    # TEST-DRIVEN EXECUTION: Golden Test 2
    test_query = "What is the Basic Structure Doctrine and how does it relate to Article 368?"
    retrieved_data = retrieve_context(test_query)
    print("\n✅ Retrieval Execution Successful. Extracted Context:\n")
    print(retrieved_data)