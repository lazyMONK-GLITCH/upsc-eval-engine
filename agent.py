import os
from dotenv import load_dotenv
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import Neo4jVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# 1. Ignite Environment Variables
load_dotenv()

# 2. Define the Cognitive State
class AgentState(TypedDict):
    query: str
    mode: str
    context: str
    final_answer: str

# 3. Ignite Core Components
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

vector_store = Neo4jVector.from_existing_index(
    embedding=embeddings,
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    index_name="vector",
    text_node_property="text",
    embedding_node_property="embedding"
)

# THE UPGRADE IS HERE: Moving from the decommissioned 3.1 to 3.3
llm = ChatGroq(
    model="llama-3.3-70b-versatile", # Core inference engine
    temperature=0.1
)

# 4. Define LangGraph Nodes
def retrieve_node(state: AgentState):
    """Fetches data from Neo4j if in query mode."""
    if state["mode"] == "query":
        # Pull the top 5 most relevant chunks from the graph
        docs = vector_store.similarity_search(state["query"], k=5)
        context = "\n\n---\n\n".join([doc.page_content for doc in docs])
        return {"context": context}
    
    # If in evaluate mode, no vector retrieval is needed
    return {"context": "Evaluation Mode Active. Vector retrieval bypassed."}

def generate_node(state: AgentState):
    """Generates the final response based on the active mode."""
    
    if state["mode"] == "evaluate":
        # THE EVALUATOR PROMPT
        prompt = PromptTemplate.from_template(
            """You are Sentinel Zero, an elite UPSC Evaluator. 
            Critically analyze the following user essay or text. 
            Identify any factual inaccuracies, structural flaws, or logical errors.
            Provide brief, brutal, and constructive feedback.
            You MUST end your response with a strict score in the exact format: X/10.
            
            User Input: {query}
            
            Evaluation:"""
        )
        chain = prompt | llm
        res = chain.invoke({"query": state["query"]})
        return {"final_answer": res.content}
        
    else:
        # THE OMNI-DISCIPLINARY QUERY PROMPT
        prompt = PromptTemplate.from_template(
            """You are Sentinel Zero, an elite UPSC Master Examiner. 
            You have access to a vast database of retrieved knowledge, including the Indian Constitution, Historical documents, and other UPSC doctrines.
            
            Use the following retrieved context to answer the user's query comprehensively and accurately. 
            If the answer is not contained in the context, state that you lack the data. Do not hallucinate outside information.
            
            Context: 
            {context}
            
            Question: {query}
            
            Answer:"""
        )
        chain = prompt | llm
        res = chain.invoke({
            "context": state["context"], 
            "query": state["query"]
        })
        return {"final_answer": res.content}

# 5. Compile the State Machine
def build_agent():
    workflow = StateGraph(AgentState)
    
    # Add Nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    
    # Define Pathways
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()