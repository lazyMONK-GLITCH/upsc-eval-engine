import streamlit as st
from agent import build_agent

# 1. Page Configuration
st.set_page_config(page_title="UPSC Beast Engine", page_icon="🦁", layout="centered")
st.title("🦁 UPSC Beast: Constitutional Evaluation Engine")
st.markdown("Powered by Neo4j Graph-RAG, Groq Llama 3.1, and LangGraph")

# 2. Initialize the LangGraph Agent in Session State
if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = build_agent()

# 3. Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. User Input and Agent Execution
if prompt := st.chat_input("Ask a question about the Indian Constitution..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Generate and display the Agent's response
    with st.chat_message("assistant"):
        with st.spinner("The Beast is searching the Constitutional Graph..."):
            try:
                # Execute the LangGraph State Machine
                final_state = st.session_state.agent_executor.invoke({"query": prompt})
                answer = final_state["final_answer"]
                
                # Display the answer
                st.markdown(answer)
                
                # Save assistant response to history
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"Engine Failure: {str(e)}")