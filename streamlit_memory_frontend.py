"""
Streamlit Memory Frontend
==========================
Production-grade UI for the AI Memory Assistant.
Supports document upload, chat interface, memory storage,
and retrieval-augmented responses.

Usage:
    streamlit run streamlit_memory_frontend.py
"""

import streamlit as st
import uuid
import queue
from pypdf import PdfReader
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# --- Backend imports ---
from memory_store import store_memory
from memory_retrieve import retrieve_memories
from memory_manager import auto_store_memories
from langgraph_unified_backend import (
    chatbot,
    retrieve_all_threads,
    submit_async_task,
)

load_dotenv()

# =====================================================================
# 1. PAGE CONFIG
# =====================================================================
st.set_page_config(
    page_title="AI Memory Assistant",
    page_icon="🧠",
    layout="wide",
)

# =====================================================================
# 2. SESSION STATE — safe initialisation (never overwrites)
# =====================================================================
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []

if "memory_loaded" not in st.session_state:
    st.session_state.memory_loaded = False

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "chat_threads" not in st.session_state:
    st.session_state.chat_threads = retrieve_all_threads()

if "thread_titles" not in st.session_state:
    st.session_state.thread_titles = {}

if "document_indexed" not in st.session_state:
    st.session_state.document_indexed = False

# LLM for title generation
_title_llm = ChatGroq(model="llama-3.3-70b-versatile")

# =====================================================================
# 3. HELPER FUNCTIONS
# =====================================================================

def generate_chat_title(message: str) -> str:
    """Generate a short descriptive title from the first user message."""
    try:
        result = _title_llm.invoke(
            [HumanMessage(content=(
                "Generate a very short title (max 5 words) for a chat that "
                "starts with this message. Return ONLY the title.\n\n"
                f"Message: {message}"
            ))]
        )
        return result.content.strip().strip('"\'')[:40]
    except Exception:
        return message[:30] + ("…" if len(message) > 30 else "")


def get_thread_title(thread_id: str) -> str:
    """Return cached title or generate from conversation."""
    titles = st.session_state.thread_titles
    if thread_id in titles and titles[thread_id] != "New Chat":
        return titles[thread_id]
    try:
        state = chatbot.get_state(
            config={"configurable": {"thread_id": thread_id}}
        )
        for msg in state.values.get("messages", []):
            if isinstance(msg, HumanMessage):
                title = generate_chat_title(msg.content)
                st.session_state.thread_titles[thread_id] = title
                return title
    except Exception:
        pass
    return f"Chat {thread_id[:8]}…"


def new_chat():
    """Start a fresh conversation thread."""
    tid = str(uuid.uuid4())
    st.session_state.thread_id = tid
    st.session_state.thread_titles[tid] = "New Chat"
    if tid not in st.session_state.chat_threads:
        st.session_state.chat_threads.append(tid)
    st.session_state.chat_history = []
    st.session_state.document_indexed = False
    st.session_state.pop("last_uploaded", None)


def load_conversation(thread_id: str) -> list:
    """Load messages from a persisted thread."""
    state = chatbot.get_state(
        config={"configurable": {"thread_id": thread_id}}
    )
    return state.values.get("messages", [])


def extract_text_from_file(uploaded_file) -> tuple:
    """Extract text from PDF, TXT, or DOCX files."""
    name = uploaded_file.name.lower()
    # --- PDF ---
    if name.endswith(".pdf"):
        try:
            reader = PdfReader(uploaded_file)
            if reader.is_encrypted:
                try:
                    reader.decrypt("")
                except Exception:
                    return None, "PDF is password-protected."
            text = "\n".join(
                p.extract_text() or "" for p in reader.pages
            )
            return text, None
        except Exception as e:
            return None, f"PDF read error: {e}"
    # --- DOCX ---
    elif name.endswith(".docx"):
        try:
            import docx
            doc = docx.Document(uploaded_file)
            text = "\n".join(p.text for p in doc.paragraphs)
            return text, None
        except ImportError:
            return None, "python-docx not installed."
        except Exception as e:
            return None, f"DOCX read error: {e}"
    # --- TXT / MD ---
    else:
        try:
            return uploaded_file.read().decode("utf-8"), None
        except Exception as e:
            return None, f"File read error: {e}"


# Ensure the current thread is tracked
if st.session_state.thread_id not in st.session_state.chat_threads:
    st.session_state.chat_threads.append(st.session_state.thread_id)

# =====================================================================
# 4. SIDEBAR
# =====================================================================
st.sidebar.title("🧠 AI Memory Assistant")
st.sidebar.divider()

# --- New Chat ---
if st.sidebar.button("➕ New Chat", use_container_width=True):
    new_chat()
    st.rerun()

# --- Document Upload ---
st.sidebar.subheader("📄 Upload Document")
uploaded_file = st.sidebar.file_uploader(
    "PDF, TXT, or DOCX",
    type=["pdf", "txt", "docx", "md"],
    label_visibility="collapsed",
)

if uploaded_file:
    already = uploaded_file.name in st.session_state.uploaded_docs
    if not already:
        with st.sidebar.status(
            f"Ingesting {uploaded_file.name}…", expanded=True
        ) as status:
            text, error = extract_text_from_file(uploaded_file)
            if error:
                status.update(label=f"❌ {error}", state="error")
            else:
                # Store document content as memory chunks
                chunks = [
                    text[i : i + 1000]
                    for i in range(0, len(text), 1000)
                    if text[i : i + 1000].strip()
                ]
                for chunk in chunks:
                    store_memory(st.session_state.user_id, chunk)
                st.session_state.uploaded_docs.append(uploaded_file.name)
                st.session_state.memory_loaded = True
                status.update(
                    label=f"✅ {uploaded_file.name} ingested ({len(chunks)} chunks)",
                    state="complete",
                )

# --- Memory Status ---
st.sidebar.divider()
st.sidebar.subheader("📊 Memory Status")
n_docs = len(st.session_state.uploaded_docs)
if st.session_state.memory_loaded or n_docs > 0:
    st.sidebar.success(f"Memory active • {n_docs} document(s) loaded")
else:
    st.sidebar.info("No documents loaded yet")

# --- Clear Memory ---
if st.sidebar.button("🗑️ Clear Memory", use_container_width=True):
    st.session_state.uploaded_docs = []
    st.session_state.memory_loaded = False
    st.sidebar.info("Memory cleared (documents removed from session)")

st.sidebar.divider()

# --- Conversation Threads ---
st.sidebar.subheader("💬 Your Chats")
for tid in st.session_state.chat_threads[::-1]:
    title = get_thread_title(tid)
    if st.sidebar.button(title, key=f"t_{tid}", use_container_width=True):
        st.session_state.thread_id = tid
        msgs = load_conversation(tid)
        st.session_state.chat_history = [
            {
                "role": "user" if isinstance(m, HumanMessage) else "assistant",
                "content": m.content,
            }
            for m in msgs
            if isinstance(m, (HumanMessage, AIMessage)) and m.content
        ]
        st.rerun()

# =====================================================================
# 5. MAIN AREA — Chat Interface
# =====================================================================
st.title("🧠 AI Memory Assistant")
st.caption("Powered by Groq LLM • Long-Term Memory • Tools • RAG")

# --- Render chat history ---
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat input ---
if st.session_state.document_indexed or st.session_state.uploaded_docs:
    placeholder = "Ask about your documents…"
else:
    placeholder = "Ask me anything…"

user_input = st.chat_input(placeholder)

if user_input:
    # Display user message immediately
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Graph config
    CONFIG = {
        "configurable": {"thread_id": st.session_state.thread_id},
        "metadata": {"thread_id": st.session_state.thread_id},
        "run_name": "memory_chat_turn",
    }

    # --- Stream assistant response ---
    with st.chat_message("assistant"):
        memory_status = st.status("🧠 Processing…", expanded=False)
        tool_holder = {"box": None}

        def ai_stream():
            eq: queue.Queue = queue.Queue()
            memory_done = False
            # Capture session values before entering the async context
            # (st.session_state is not available on background threads)
            _user_id = st.session_state.user_id

            async def _run():
                try:
                    async for chunk, meta in chatbot.astream(
                        {
                            "messages": [HumanMessage(content=user_input)],
                            "user_id": _user_id,
                        },
                        config=CONFIG,
                        stream_mode="messages",
                    ):
                        eq.put((chunk, meta))
                except Exception as exc:
                    eq.put(("error", exc))
                finally:
                    eq.put(None)

            submit_async_task(_run())

            while True:
                item = eq.get()
                if item is None:
                    break
                if item[0] == "error":
                    yield f"\n\n⚠️ Error: {item[1]}"
                    break

                chunk, meta = item
                node = meta.get("langgraph_node", "")

                # Skip memory-internal LLM output
                if node in ("memory_extraction", "memory_retrieval"):
                    continue

                is_chat = node == "chat_node"

                # Transition memory status
                if is_chat and isinstance(chunk, AIMessage) and not memory_done:
                    memory_status.update(
                        label="✅ Memories loaded",
                        state="complete",
                        expanded=False,
                    )
                    memory_done = True

                # Tool status
                if isinstance(chunk, ToolMessage):
                    name = getattr(chunk, "name", "tool")
                    if tool_holder["box"] is None:
                        tool_holder["box"] = st.status(
                            f"🔧 Using `{name}` …", expanded=True
                        )
                    else:
                        tool_holder["box"].update(
                            label=f"🔧 Using `{name}` …",
                            state="running",
                            expanded=True,
                        )

                # Yield only chat-node AI tokens
                if is_chat and isinstance(chunk, AIMessage) and chunk.content:
                    yield chunk.content

            if not memory_done:
                memory_status.update(
                    label="✅ Done", state="complete", expanded=False
                )

        ai_message = st.write_stream(ai_stream())

        # Finalise tool status
        if tool_holder["box"] is not None:
            tool_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    # Save assistant response
    if ai_message:
        st.session_state.chat_history.append(
            {"role": "assistant", "content": ai_message}
        )

    # Generate title on first message
    tid = st.session_state.thread_id
    if (
        tid not in st.session_state.thread_titles
        or st.session_state.thread_titles.get(tid) == "New Chat"
    ):
        st.session_state.thread_titles[tid] = generate_chat_title(user_input)

    # Refresh UI
    st.rerun()
