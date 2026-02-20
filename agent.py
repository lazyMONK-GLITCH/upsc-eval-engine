import os
from typing import TypedDict, List
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# Import our custom Brain and Eyes
from router import route_user_query
from retriever import retrieve_context

load_dotenv()

# 1. Define the Digital Physics of the Agent's Memory (State)
class AgentState(TypedDict):
    query: str
    intent: str
    entities: List[str]
    context: str
    final_answer: str

# 2. Define the Discrete Execution Nodes
def node_route(state: AgentState):
    print("🟢 LANGGRAPH NODE: Routing Query...")
    route = route_user_query(state["query"])
    return {"intent": route.intent, "entities": route.entities}

def node_retrieve(state: AgentState):
    print("🟢 LANGGRAPH NODE: Retrieving Context from Neo4j...")
    context = retrieve_context(state["query"])
    return {"context": context}

def node_generate(state: AgentState):
    print("🟢 LANGGRAPH NODE: Generating UPSC Grade Answer...")
    
    # Using Llama 3.1 8B via Groq for high-speed, zero-cost inference
    llm = ChatGroq(
        temperature=0.2, # Low temperature for factual strictness
        model_name="llama-3.1-8b-instant", 
        api_key=os.getenv("GROQ_API_KEY")
    )
    
    system_prompt = f"""
    You are an elite UPSC Evaluation Engine. 
    Your strict directive is to answer the user's query utilizing ONLY the provided Constitutional Context retrieved from the database.
    If the context is insufficient, explicitly state that you lack the data. Do not hallucinate case laws.
    
    --- CONTEXT BLOCK ---
    {state['context']}
    ---------------------
    """
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["query"])
    ])
    
    return {"final_answer": response.content}

# 3. Build the Directed Graph Workflow
def build_agent():
    workflow = StateGraph(AgentState)
    
    # Add Nodes
    workflow.add_node("route", node_route)
    workflow.add_node("retrieve", node_retrieve)
    workflow.add_node("generate", node_generate)
    
    # Define the strict flow of logic
    workflow.set_entry_point("route")
    workflow.add_edge("route", "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()

if __name__ == "__main__":
    print("🚀 INITIALIZING UPSC BEAST AGENT 🚀\n")
    
    agent_executor = build_agent()
    
    # TEST-DRIVEN EXECUTION: The Final Pipeline Test
    test_query = "What constitutes the basic features of the Constitution based on recent Supreme Court judgements?"
    
    print(f"USER QUERY: {test_query}\n")
    
    # Execute the LangGraph State Machine
    final_state = agent_executor.invoke({"query": test_query})
    
    print("\n" + "="*60)
    print("FINAL UPSC ANSWER:")
    print("="*60)
    print(final_state["final_answer"])