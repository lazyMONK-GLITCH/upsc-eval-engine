import os
from typing import TypedDict, List
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from router import route_user_query
from retriever import retrieve_context

load_dotenv()

# 1. State Definition (Notice the new 'mode' parameter)
class AgentState(TypedDict):
    query: str       
    mode: str        # 'query' or 'evaluate'
    intent: str
    entities: List[str]
    context: str     
    final_answer: str 

# 2. The Cognitive Nodes
def node_route(state: AgentState):
    print(f"🟢 LANGGRAPH NODE: Routing ({state.get('mode', 'query')} mode)...")
    route = route_user_query(state["query"])
    return {"intent": route.intent, "entities": route.entities}

def node_retrieve(state: AgentState):
    print("🟢 LANGGRAPH NODE: Retrieving Absolute Truth from Neo4j...")
    context = retrieve_context(state["query"])
    return {"context": context}

def node_generate(state: AgentState):
    print("🟢 LANGGRAPH NODE: Generating Output...")
    
    llm = ChatGroq(
        temperature=0.1, 
        model_name="llama-3.1-8b-instant", 
        api_key=os.getenv("GROQ_API_KEY")
    )
    
    # DYNAMIC COGNITIVE SWITCH
    if state.get("mode") == "evaluate":
        system_prompt = f"""
        You are Sentinel Zero, an elite, strict UPSC Mains Examiner evaluating a student's answer.
        Evaluate the student's payload utilizing ONLY the provided Constitutional Context retrieved from the Neo4j Graph Database.
        
        Provide your evaluation in the following strict markdown format:
        ### 📊 Final UPSC Score: [Insert Score]/10
        **1. Factual Accuracy:** [Evaluate alignment with retrieved context]
        **2. Missing Constitutional Elements:** [Identify exact missing Articles/Cases]
        **3. Structure & Presentation:** [Critique flow and structure]
        
        --- CONTEXT BLOCK (Absolute Truth) ---
        {state['context']}
        --------------------------------------
        """
        user_payload = f"Student Answer Payload:\n\n{state['query']}"
        
    else:
        system_prompt = f"""
        You are Sentinel Zero, an elite Constitutional Intelligence Engine.
        Your strict directive is to answer the user's query utilizing ONLY the provided Constitutional Context retrieved from the database.
        If the context is insufficient, explicitly state that you lack the data. Do not hallucinate.
        
        --- CONTEXT BLOCK ---
        {state['context']}
        ---------------------
        """
        user_payload = state["query"]
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_payload)
    ])
    
    return {"final_answer": response.content}

# 3. Build the Directed Graph Workflow
def build_agent():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("route", node_route)
    workflow.add_node("retrieve", node_retrieve)
    workflow.add_node("generate", node_generate)
    
    workflow.set_entry_point("route")
    workflow.add_edge("route", "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()