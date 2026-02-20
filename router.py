import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from schemas import RouteQuery

load_dotenv()

def route_user_query(query: str) -> RouteQuery:
    print(f"🚦 Activating Groq Router for query: '{query}'")
    
    # Enforcing Zero-Cost Constraint: Groq Llama-3.1 handles intent extraction
    llm = ChatGroq(
        temperature=0, 
        model_name="llama-3.1-8b-instant", 
        api_key=os.getenv("GROQ_API_KEY")
    )
    
    # We bind the LLM strictly to our Pydantic Digital Physics
    structured_router = llm.with_structured_output(RouteQuery)
    
    system_prompt = """
    You are an expert UPSC Evaluation Engine routing mechanism.
    Your strictly limited job is to analyze the user's input, determine their analytical intent, and extract exact entities for a Neo4j database vector search.
    """
    
    # Execute deterministic inference
    result = structured_router.invoke(f"{system_prompt}\n\nUser Query: {query}")
    return result

if __name__ == "__main__":
    # TEST-DRIVEN EXECUTION: Golden Test 1
    test_query = "Critically analyze the Basic Structure Doctrine established in the Kesavananda Bharati case."
    route = route_user_query(test_query)
    print("\n✅ Router Execution Successful:")
    print(f"Extracted Intent: {route.intent}")
    print(f"Extracted Entities: {route.entities}")