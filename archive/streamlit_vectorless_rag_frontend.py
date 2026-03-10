import streamlit as st
import queue
import uuid
import io
from pypdf import PdfReader
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph_vectorless_rag_backend import chatbot, retrieve_all_threads, submit_async_task, llm

# =========================== Utilities ===========================
def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    # Default title until first message is sent
    st.session_state.setdefault("thread_titles", {})[thread_id] = "New Chat"
    add_thread(thread_id)
    st.session_state["message_history"] = []
    st.session_state["document_indexed"] = False

def add_thread(thread_id):
    if "chat_threads" not in st.session_state:
        st.session_state["chat_threads"] = []
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])

def generate_title_from_message(message: str) -> str:
    """Use LLM to generate a short, meaningful title from the first user message."""
    try:
        prompt = (
            f"Generate a very short title (3-6 words, no quotes) that summarizes this message: {message[:200]}"
        )
        response = llm.invoke(prompt)
        title = response.content.strip().strip('"').strip("'")
        # Truncate if too long
        return title[:40] if len(title) > 40 else title
    except Exception:
        # Fallback: first 40 chars of the message
        return message[:40].strip() + ("..." if len(message) > 40 else "")

def extract_text_from_file(uploaded_file):
    """Extract text from PDF, TXT, or MD files. Handles encrypted PDFs gracefully."""
    if uploaded_file.type == "application/pdf":
        try:
            reader = PdfReader(uploaded_file)
            # Try to decrypt if the PDF is encrypted
            if reader.is_encrypted:
                try:
                    reader.decrypt("")  # Try empty password first
                except Exception:
                    return None, "This PDF is password-protected. Please provide an unprotected PDF."
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            return text, None
        except Exception as e:
            return None, f"Failed to read PDF: {str(e)}"
    else:
        try:
            content = uploaded_file.read().decode("utf-8")
            return content, None
        except Exception as e:
            return None, f"Failed to read file: {str(e)}"

# ======================= Session Initialization ===================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    existing_threads = retrieve_all_threads()
    st.session_state["chat_threads"] = existing_threads

if "thread_titles" not in st.session_state:
    # Initialise titles with UUID shortnames for existing threads
    st.session_state["thread_titles"] = {}

if "document_indexed" not in st.session_state:
    st.session_state["document_indexed"] = False

add_thread(st.session_state["thread_id"])

# ============================ Sidebar ============================
st.sidebar.title("🧠 Vectorless RAG")
st.sidebar.info("Using PageIndex MCP tools for indexing and search.")

if st.sidebar.button("➕  New Chat"):
    reset_chat()
    st.rerun()

st.sidebar.header("My Conversations")
for thread_id in st.session_state["chat_threads"][::-1]:
    # Show user-friendly title or a truncated UUID fallback
    title = st.session_state["thread_titles"].get(thread_id)
    if not title:
        title = f"Chat {thread_id[:8]}…"
    if st.sidebar.button(title, key=f"btn_{thread_id}"):
        st.session_state["thread_id"] = thread_id
        messages = load_conversation(thread_id)
        temp_messages = []
        for msg in messages:
            if isinstance(msg, (HumanMessage, AIMessage)) and msg.content:
                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                temp_messages.append({"role": role, "content": msg.content})
        st.session_state["message_history"] = temp_messages
        # Re-detect if a document was indexed in this thread
        st.session_state["document_indexed"] = any(
            m["role"] == "assistant" and "indexed" in m["content"].lower()
            for m in temp_messages
        )
        st.session_state.pop("last_uploaded", None)
        st.rerun()

# ============================ Main UI =============================
st.title("Vectorless RAG Assistant")

# Render history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# File Uploader
with st.container():
    uploaded_file = st.file_uploader(
        "Upload Document for Indexing",
        type=["pdf", "txt", "md"],
        label_visibility="collapsed"
    )
    if uploaded_file:
        if "last_uploaded" not in st.session_state or st.session_state["last_uploaded"] != uploaded_file.name:
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

                    CONFIG = {"configurable": {"thread_id": st.session_state["thread_id"]}}
                    event_queue = queue.Queue()

                    async def run_indexing():
                        try:
                            async for chunk, _ in chatbot.astream(
                                {"messages": [HumanMessage(content=indexing_prompt)]},
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
                            status.write(f"🔧 Using tools for indexing...")

                    if final_response:
                        st.session_state["message_history"].append(
                            {"role": "assistant", "content": final_response}
                        )

                    status.update(label="✅ Indexing Complete", state="complete")
                    st.session_state["last_uploaded"] = uploaded_file.name
                    st.session_state["document_indexed"] = True
                    st.rerun()

# Dynamic chat input placeholder
if st.session_state.get("document_indexed") or uploaded_file:
    placeholder_text = "Ask about your documents..."
else:
    placeholder_text = "Ask me anything..."

user_input = st.chat_input(placeholder_text)

if user_input:
    # Generate a title from the first message in this thread
    is_first_message = len(st.session_state["message_history"]) == 0
    current_thread = st.session_state["thread_id"]

    if is_first_message and current_thread not in st.session_state["thread_titles"]:
        title = generate_title_from_message(user_input)
        st.session_state["thread_titles"][current_thread] = title

    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
    }

    with st.chat_message("assistant"):
        status_holder = {"box": None}

        def ai_only_stream():
            event_queue: queue.Queue = queue.Queue()

            async def run_stream():
                try:
                    async for message_chunk, _ in chatbot.astream(
                        {"messages": [HumanMessage(content=user_input)]},
                        config=CONFIG,
                        stream_mode="messages",
                    ):
                        event_queue.put(message_chunk)
                except Exception as e:
                    event_queue.put(("error", str(e)))
                finally:
                    event_queue.put(None)

            submit_async_task(run_stream())

            while True:
                item = event_queue.get()
                if item is None:
                    break

                if isinstance(item, tuple) and item[0] == "error":
                    yield f"\n\n⚠️ Error: {item[1]}"
                    break

                if isinstance(item, ToolMessage):
                    tool_name = getattr(item, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(f"🔧 Using `{tool_name}` ...", expanded=True)
                    else:
                        status_holder["box"].update(label=f"🔧 Using `{tool_name}` ...")

                if isinstance(item, AIMessage) and item.content:
                    yield item.content

        ai_message = st.write_stream(ai_only_stream())
        if status_holder["box"]:
            status_holder["box"].update(label="✅ Tool finished", state="complete", expanded=False)

    if ai_message:
        st.session_state["message_history"].append({"role": "assistant", "content": ai_message})
