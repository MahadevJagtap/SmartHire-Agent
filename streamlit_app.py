"""
Unified Streamlit Frontend
============================
Single frontend combining ALL features:
  • Streaming responses
  • Conversation threads with LLM-generated titles
  • Tool usage status indicators
  • MCP (async) tool support
  • PDF / TXT / MD document upload & indexing (Vectorless RAG)
  • Long-term memory (user profile, stored memories sidebar)
  • Database persistence across restarts
"""

import streamlit as st
import queue
import uuid
import os
from pypdf import PdfReader
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph_unified_backend import (
    chatbot,
    retrieve_all_threads,
    submit_async_task,
    llm,
)
from recruitment_view import recruitment_page

from dotenv import load_dotenv

load_dotenv()

# ============================= Page Config ================================
st.set_page_config(
    page_title="AI Agent — Unified",
    page_icon="🤖",
    layout="wide",
)

# ============================= Custom CSS =================================
st.markdown("""
<style>
    /* Clean sidebar styling */
    [data-testid="stSidebar"] { min-width: 280px; }
</style>
""", unsafe_allow_html=True)

# ============================= Utilities ==================================

_title_llm = ChatGroq(model="llama-3.3-70b-versatile")


def generate_thread_id():
    return str(uuid.uuid4())


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    st.session_state.setdefault("thread_titles", {})[thread_id] = "New Chat"
    add_thread(thread_id)
    st.session_state["message_history"] = []
    st.session_state["document_indexed"] = False
    st.session_state.pop("last_uploaded", None)


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


def generate_chat_title(user_message: str) -> str:
    """Use the LLM to generate a short, descriptive title for the chat."""
    try:
        result = _title_llm.invoke(
            [HumanMessage(content=(
                "Generate a very short title (max 5 words) for a chat that starts with "
                "this user message. Return ONLY the title, nothing else.\n\n"
                f"User message: {user_message}"
            ))]
        )
        title = result.content.strip().strip('"').strip("'")
        return title[:40]
    except Exception:
        return user_message[:30] + ("…" if len(user_message) > 30 else "")


def get_thread_title(thread_id: str) -> str:
    """Get a cached title for a thread, or generate one from its first message."""
    titles = st.session_state.get("thread_titles", {})
    if thread_id in titles:
        return titles[thread_id]

    # Derive title from the first user message
    try:
        messages = load_conversation(thread_id)
        for msg in messages:
            if isinstance(msg, HumanMessage):
                title = generate_chat_title(msg.content)
                st.session_state["thread_titles"][thread_id] = title
                return title
    except Exception:
        pass

    return f"Chat {str(thread_id)[:8]}…"


def extract_text_from_file(uploaded_file):
    """Extract text from PDF, TXT, or MD files."""
    if uploaded_file.type == "application/pdf":
        try:
            reader = PdfReader(uploaded_file)
            if reader.is_encrypted:
                try:
                    reader.decrypt("")
                except Exception:
                    return None, "This PDF is password-protected."
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            return text, None
        except Exception as e:
            return None, f"Failed to read PDF: {e}"
    else:
        try:
            return uploaded_file.read().decode("utf-8"), None
        except Exception as e:
            return None, f"Failed to read file: {e}"


# ========================= Session Initialization =========================

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "user_id" not in st.session_state:
    st.session_state["user_id"] = "default_user"

if "thread_titles" not in st.session_state:
    st.session_state["thread_titles"] = {}

if "document_indexed" not in st.session_state:
    st.session_state["document_indexed"] = False

add_thread(st.session_state["thread_id"])

# ================================ Sidebar =================================

st.sidebar.title("🤖 AI Agent")
st.sidebar.divider()

# --- Navigation ---
st.sidebar.subheader("🚀 Mode")
page = st.sidebar.radio("Select Page", ["Chatbot", "Recruitment Agent"], label_visibility="collapsed")

st.sidebar.divider()

if page == "Chatbot":
    # --- New Chat ---
    if st.sidebar.button("➕ New Chat", use_container_width=True):
        reset_chat()
        st.rerun()

# --- File upload (Vectorless RAG) ---
st.sidebar.subheader("📄 Upload Document")
uploaded_file = st.sidebar.file_uploader(
    "PDF, TXT, or MD",
    type=["pdf", "txt", "md"],
    label_visibility="collapsed",
)

st.sidebar.divider()

# --- Conversation threads ---
st.sidebar.subheader("💬 Your Chats")
for thread_id in st.session_state["chat_threads"][::-1]:
    title = get_thread_title(thread_id)
    if st.sidebar.button(title, key=f"thread_{thread_id}", use_container_width=True):
        st.session_state["thread_id"] = thread_id
        messages = load_conversation(thread_id)
        temp_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                temp_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage) and msg.content:
                temp_messages.append({"role": "assistant", "content": msg.content})
        st.session_state["message_history"] = temp_messages
        # Detect if a document was indexed in this thread
        st.session_state["document_indexed"] = any(
            m["role"] == "assistant" and "indexed" in m["content"].lower()
            for m in temp_messages
        )
        st.session_state.pop("last_uploaded", None)
        st.rerun()

# ============================== Main Logic =================================

if page == "Chatbot":
    st.title("🤖 AI Agent")
    st.caption("Powered by Groq LLM • Memory • Tools • RAG")

    # Render chat history
    for message in st.session_state["message_history"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # ============================== File Indexing ==============================

    if uploaded_file:
        if (
            "last_uploaded" not in st.session_state
            or st.session_state["last_uploaded"] != uploaded_file.name
        ):
            with st.status(f"Indexing {uploaded_file.name}...", expanded=True) as status:
                text, error = extract_text_from_file(uploaded_file)
                if error:
                    status.update(label=f"❌ {error}", state="error")
                    st.error(error)
                else:
                    indexing_prompt = (
                        f"Please index the document '{uploaded_file.name}'. "
                        f"Content:\n\n{text[:10000]}"
                    )
                    CONFIG = {
                        "configurable": {"thread_id": st.session_state["thread_id"]}
                    }
                    event_queue = queue.Queue()
                    _idx_user_id = st.session_state["user_id"]

                    async def run_indexing():
                        try:
                            async for chunk, _ in chatbot.astream(
                                {
                                    "messages": [HumanMessage(content=indexing_prompt)],
                                    "user_id": _idx_user_id,
                                },
                                config=CONFIG,
                                stream_mode="messages",
                            ):
                                event_queue.put(chunk)
                        except Exception as e:
                            event_queue.put(("error", str(e)))
                        finally:
                            event_queue.put(None)

                    submit_async_task(run_indexing())

                    final_response = ""
                    while True:
                        chunk = event_queue.get()
                        if chunk is None:
                            break
                        if isinstance(chunk, tuple) and chunk[0] == "error":
                            st.error(f"Indexing error: {chunk[1]}")
                            break
                        if isinstance(chunk, AIMessage) and chunk.content:
                            final_response = chunk.content
                        elif isinstance(chunk, ToolMessage):
                            status.write("🔧 Using tools for indexing...")

                    if final_response:
                        st.session_state["message_history"].append(
                            {"role": "assistant", "content": final_response}
                        )

                    status.update(label="✅ Indexing Complete", state="complete")
                    st.session_state["last_uploaded"] = uploaded_file.name
                    st.session_state["document_indexed"] = True
                    st.rerun()

    # ============================== Chat Input ================================

    if st.session_state.get("document_indexed") or uploaded_file:
        placeholder = "Ask about your documents..."
    else:
        placeholder = "Ask me anything..."

    user_input = st.chat_input(placeholder)

    if user_input:
        # Show user message
        st.session_state["message_history"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        CONFIG = {
            "configurable": {"thread_id": st.session_state["thread_id"]},
            "metadata": {"thread_id": st.session_state["thread_id"]},
            "run_name": "unified_chat_turn",
        }

        # Assistant response with memory + tool status
        with st.chat_message("assistant"):
            memory_status = st.status("🧠 Processing memories…", expanded=False)
            status_holder = {"box": None}

            def ai_stream():
                event_queue: queue.Queue = queue.Queue()
                memory_phase_done = False
                # Capture before entering async context
                _user_id = st.session_state["user_id"]

                async def run_stream():
                    try:
                        async for message_chunk, metadata in chatbot.astream(
                            {
                                "messages": [HumanMessage(content=user_input)],
                                "user_id": _user_id,
                            },
                            config=CONFIG,
                            stream_mode="messages",
                        ):
                            event_queue.put((message_chunk, metadata))
                    except Exception as exc:
                        event_queue.put(("error", exc))
                    finally:
                        event_queue.put(None)

                submit_async_task(run_stream())

                while True:
                    item = event_queue.get()
                    if item is None:
                        break

                    if item[0] == "error":
                        yield f"\n\n⚠️ Error: {item[1]}"
                        break

                    message_chunk, metadata = item
                    node = metadata.get("langgraph_node", "")

                    # Skip extraction/retrieval LLM output
                    if node in ("memory_extraction", "memory_retrieval"):
                        continue

                    is_chat = node == "chat_node"

                    # Update memory status on first chat node message
                    if is_chat and isinstance(message_chunk, AIMessage) and not memory_phase_done:
                        memory_status.update(
                            label="✅ Memories loaded", state="complete", expanded=False
                        )
                        memory_phase_done = True

                    # Tool status indicator
                    if isinstance(message_chunk, ToolMessage):
                        tool_name = getattr(message_chunk, "name", "tool")
                        if status_holder["box"] is None:
                            status_holder["box"] = st.status(
                                f"🔧 Using `{tool_name}` …", expanded=True
                            )
                        else:
                            status_holder["box"].update(
                                label=f"🔧 Using `{tool_name}` …",
                                state="running",
                                expanded=True,
                            )

                    # Stream only chat node AI content
                    if is_chat and isinstance(message_chunk, AIMessage) and message_chunk.content:
                        yield message_chunk.content

                if not memory_phase_done:
                    memory_status.update(
                        label="✅ Done", state="complete", expanded=False
                    )

            ai_message = st.write_stream(ai_stream())

            # Finalize tool status
            if status_holder["box"] is not None:
                status_holder["box"].update(
                    label="✅ Tool finished", state="complete", expanded=False
                )

        # Save assistant response
        if ai_message:
            st.session_state["message_history"].append(
                {"role": "assistant", "content": ai_message}
            )

        # Generate title after first message in a new chat
        thread_id = st.session_state["thread_id"]
        if thread_id not in st.session_state["thread_titles"] or st.session_state["thread_titles"].get(thread_id) == "New Chat":
            st.session_state["thread_titles"][thread_id] = generate_chat_title(user_input)

        # Refresh sidebar
        st.rerun()
else:
    recruitment_page()
