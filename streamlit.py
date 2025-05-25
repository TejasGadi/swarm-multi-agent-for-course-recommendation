from src.config.config import OPENAI_API_KEY, TAVILY_API_KEY, EXA_API_KEY, PINECONE_API_KEY, LLM_MODEL
import os
from pathlib import Path
from src.agents import student_profile_agent, course_discovery_agent, career_path_agent
from src.reset_profile import reset_profile_to_empty

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
os.environ["EXA_API_KEY"] = EXA_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["LLM_MODEL"] = LLM_MODEL

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

# short-term memory
checkpointer = InMemorySaver()
# long-term memory
store = InMemoryStore()


from langgraph_swarm import create_swarm
from langgraph_swarm import SwarmState

# --- Swarm Creation ---
swarm = create_swarm(
    agents=[
        student_profile_agent,
        course_discovery_agent,
        career_path_agent
    ],
    default_active_agent="student_profile_agent",
).compile(
    checkpointer=checkpointer,
    store=store
)

# Delete
store.delete(("swarm",), "1")
checkpointer.delete_thread("1")

# Profile reset to empty values on starting chat
reset_profile_to_empty()

config = {"configurable": {"thread_id": "3"}}


profile_json_path = Path("./profile.json")

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

st.set_page_config(page_title="LangGraph Chat", page_icon="🧠")

st.title("🧠 LangGraph Multi-Agent Chat")
st.markdown("Talk to your multi-agent system. Tool calls and handoffs included!")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_message_count" not in st.session_state:
    st.session_state.last_message_count = 0


# Input area
user_input = st.chat_input("Say something...")

if user_input:
    # Append user's message to chat history
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    # Run a single LangGraph turn
    result = swarm.invoke({"messages": st.session_state.chat_history}, config)

    # Get updated messages
    updated_messages = result.get("messages", [])

    # Determine new messages since last turn
    new_messages = updated_messages[len(st.session_state.chat_history):]

    # Append only new messages to the history
    st.session_state.chat_history.extend(new_messages)
    st.session_state.last_message_count = len(st.session_state.chat_history)

# Display messages
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)
            for tool_call in msg.tool_calls or []:
                st.markdown(
                    f"🔧 **Tool Call**\n\n- Name: `{tool_call['name']}`\n- ID: `{tool_call['id']}`\n- Args: `{tool_call['args']}`",
                    unsafe_allow_html=True,
                )
    elif isinstance(msg, ToolMessage):
        with st.chat_message("tool"):
            st.markdown(
                f"🛠️ **Tool Response**\n\n- ID: `{msg.tool_call_id}`\n- Result: `{msg.content}`",
                unsafe_allow_html=True,
            )
    else:
        with st.chat_message("system"):
            st.markdown(f"📦 {msg.type}: {msg.content}")