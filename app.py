import streamlit as st
from agent import build_agent

# 1. Elegant Page Configuration (Sidebar expanded by default now)
st.set_page_config(page_title="Sentinel Zero", page_icon="💠", layout="centered", initial_sidebar_state="expanded")

# 2. CSS Injection for Premium Aesthetics
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    /* header visibility restored for mobile sidebar toggle */
    
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
    .streamlit-expanderHeader { font-size: 0.85rem; color: #888; border-bottom: 1px solid #333; }
    .stChatInputContainer { border-radius: 20px; }
</style>
""", unsafe_allow_html=True)

# --- NEW: COGNITIVE CONTROL SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Cognitive Core")
    engine_mode = st.radio(
        "Select Engine Directive:",
        ["🔍 Query Database", "📊 Evaluate Answer"],
        help="Query: Asks the engine a question. Evaluate: Grades a written essay."
    )
    st.divider()
    st.caption("Neo4j Vector Retrieval: ACTIVE")
    st.caption("Groq Llama 3.1 Inference: ACTIVE")

# Set the internal state flag based on UI selection
mode_flag = "evaluate" if engine_mode == "📊 Evaluate Answer" else "query"

st.title("💠 SENTINEL ZERO")
st.caption("Cognitive Auditing Core | Llama 3.1 x Neo4j")

# 3. Initialize the LangGraph State Machine
if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = build_agent()

# 4. Initialize the Chat Memory
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Sentinel Zero online. Select directive in sidebar and enter data."}]

# 5. Render Historical Chat
for message in st.session_state.messages:
    avatar = "💠" if message["role"] == "assistant" else "👤"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        if "telemetry" in message and message["telemetry"]:
            with st.expander("🔍 View Engine Telemetry (Neo4j Graph Retrieval)"):
                st.code(message["telemetry"], language="markdown")

# 6. The Execution Loop
if prompt := st.chat_input(f"Enter text for {engine_mode}..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
        
    with st.chat_message("assistant", avatar="💠"):
        answer = None
        telemetry_data = None
        
        with st.status(f"🧠 Executing {engine_mode} via Neo4j...", expanded=True) as status:
            try:
                # PASING THE MODE FLAG TO LANGGRAPH
                final_state = st.session_state.agent_executor.invoke({
                    "query": prompt,
                    "mode": mode_flag
                })
                answer = final_state.get("final_answer", "No answer generated.")
                telemetry_data = final_state.get("context", "No context retrieved.")
                
                st.markdown("**Retrieved Vectors:**")
                st.text(telemetry_data)
                status.update(label="✅ Graph Execution Complete", state="complete", expanded=False)
                
            except Exception as e:
                status.update(label="❌ Engine Failure", state="error", expanded=True)
                st.error(f"Terminal Error: {str(e)}")

        if answer:
            st.markdown(answer)
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer,
                "telemetry": telemetry_data
            })