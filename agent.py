import os
from typing import TypedDict, List
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from router import route_user_query
from retriever import retrieve_context

load_dotenv()

# 1. State Definition
class AgentState(TypedDict):
    query: str       
    intent: str
    entities: List[str]
    context: str     
    final_answer: str 

# 2. The Cognitive Nodes
def node_route(state: AgentState):
    print("🟢 LANGGRAPH NODE: Analyzing Essay Intent...")
    route = route_user_query(state["query"])
    return {"intent": route.intent, "entities": route.entities}

def node_retrieve(state: AgentState):
    print("🟢 LANGGRAPH NODE: Retrieving Absolute Truth from Neo4j...")
    context = retrieve_context(state["query"])
    return {"context": context}

def node_generate(state: AgentState):
    print("🟢 LANGGRAPH NODE: Generating UPSC Evaluation Report...")
    
    llm = ChatGroq(
        temperature=0.1, 
        model_name="llama-3.1-8b-instant", 
        api_key=os.getenv("GROQ_API_KEY")
    )
    
    system_prompt = f"""
    You are Sentinel Zero, an elite, strict UPSC Mains Examiner evaluating a GS Paper 2 (Polity) student answer.
    
    Your strict directive is to evaluate the student's payload utilizing ONLY the provided Constitutional Context retrieved from the Neo4j Graph Database.
    Compare their answer against the absolute truth in the database. Do not hallucinate external facts.
    
    Provide your evaluation in the following strict markdown format:
    
    ### 📊 Final UPSC Score: [Insert Score]/10
    
    **1. Factual Accuracy:**
    [Evaluate if what they wrote aligns with the retrieved Constitutional Context. Point out any specific factual errors or misinterpretations.]
    
    **2. Missing Constitutional Elements:**
    [Identify exact Articles, Clauses, or Supreme Court Case Laws that are present in the retrieved context but the student failed to mention.]
    
    **3. Structure & Presentation:**
    [Critique the flow, introduction, and conclusion as per standard UPSC expectations.]
    
    --- CONTEXT BLOCK (Absolute Truth) ---
    {state['context']}
    --------------------------------------
    """
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Student Answer Payload:\n\n{state['query']}")
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