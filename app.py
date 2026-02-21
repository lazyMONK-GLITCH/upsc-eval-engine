import streamlit as st
from agent import build_agent

# 1. Elegant Page Configuration (Minimalist Mode)
st.set_page_config(
    page_title="Sentinel Zero", 
    page_icon="◈", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 2. CSS Injection for Premium Aesthetics
st.markdown("""
<style>
    /* Strip default Streamlit artifacts */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Modern Typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    
    /* Clean up the expander UI */
    .streamlit-expanderHeader {
        font-size: 0.85rem;
        color: #888;
        border-bottom: 1px solid #333;
    }
    
    /* Custom input box styling */
    .stChatInputContainer {
        border-radius: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.title("◈ SENTINEL ZERO")
st.caption("Cognitive Auditing Core | Llama 3.1 x Neo4j")

# 3. Initialize the LangGraph State Machine
if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = build_agent()

# 4. Initialize the Chat Memory
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Sentinel Zero online. Awaiting data parameters."}]

# 5. Render Historical Chat
for message in st.session_state.messages:
    # Custom Avatars
    avatar = "💠" if message["role"] == "assistant" else "👤"
    
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        
        # If the engine saved telemetry for this message, render the hidden dropdown
        if "telemetry" in message and message["telemetry"]:
            with st.expander("🔍 View Engine Telemetry (Neo4j Graph Retrieval)"):
                st.code(message["telemetry"], language="markdown")

# 6. The Execution Loop
if prompt := st.chat_input("Input query or text payload for evaluation..."):
    # Render User Input
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
        
    # Render Assistant Response
    with st.chat_message("assistant", avatar="💠"):
        answer = None
        telemetry_data = None
        
        # The Dynamic Gemini-style Status Box
        with st.status("🧠 Querying Neo4j Vector Database...", expanded=True) as status:
            try:
                # Execute the LangGraph workflow
                final_state = st.session_state.agent_executor.invoke({"query": prompt})
                answer = final_state.get("final_answer", "No answer generated.")
                telemetry_data = final_state.get("context", "No context retrieved.")
                
                # Display the raw data inside the status box temporarily
                st.markdown("**Retrieved Vectors:**")
                st.text(telemetry_data)
                
                # Collapse the status box and change to a success state
                status.update(label="✅ Graph Search Complete", state="complete", expanded=False)
                
            except Exception as e:
                status.update(label="❌ Engine Failure", state="error", expanded=True)
                st.error(f"Terminal Error: {str(e)}")

        # Render the final generated answer clearly below the collapsed status box
        if answer:
            st.markdown(answer)
            # Save both the final answer and the raw telemetry to memory
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer,
                "telemetry": telemetry_data
            })