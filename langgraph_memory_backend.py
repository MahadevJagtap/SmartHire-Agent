"""
LangGraph Backend with Long-Term Memory
========================================
Integrates the modular memory system into a LangGraph StateGraph so the agent can:
1. Automatically extract and store user facts
2. Retrieve relevant memories before responding
3. Generate a memory-augmented response via Groq LLM
"""

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from memory_manager import (
    get_relevant_memories,
    auto_store_memories,
    get_user_memories,
)
import sqlite3
import logging
import os

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LangSmith tracing
# ---------------------------------------------------------------------------
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "chatbot-langgraph")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv(
    "LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"
)

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
llm = ChatGroq(model="llama-3.3-70b-versatile")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str
    retrieved_memories: list[str]


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------
SYSTEM_TEMPLATE = """You are an AI assistant with long-term memory.
Use the following stored memories about this user when relevant to personalize your responses.
Do NOT explicitly tell the user you are "storing" or "retrieving" memories — just use the knowledge naturally.

{memories_section}"""


def _build_system_prompt(memories: list[str]) -> str:
    if memories:
        bullets = "\n".join(f"• {m}" for m in memories)
        memories_section = f"User memories:\n{bullets}"
    else:
        memories_section = "No stored memories for this user yet."
    return SYSTEM_TEMPLATE.format(memories_section=memories_section)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def memory_extraction_node(state: ChatState) -> dict:
    """Extract personal facts from the latest user message and store them."""
    user_id = state.get("user_id", "default_user")
    messages = state["messages"]

    # Find the latest human message
    latest_human_msg = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            latest_human_msg = msg.content
            break

    if not latest_human_msg:
        return {}

    try:
        stored = auto_store_memories(user_id, latest_human_msg)
        if stored:
            logger.info(
                "Auto-extracted %d new memories for user '%s'", len(stored), user_id
            )
    except Exception as e:
        logger.error("Memory extraction failed: %s", e)

    return {}


def memory_retrieval_node(state: ChatState) -> dict:
    """Retrieve relevant memories based on the latest user message."""
    user_id = state.get("user_id", "default_user")
    messages = state["messages"]

    # Find the latest human message
    latest_human_msg = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            latest_human_msg = msg.content
            break

    if not latest_human_msg:
        return {"retrieved_memories": []}

    try:
        results = get_relevant_memories(user_id, latest_human_msg, k=5)
        memories = [r["content"] for r in results if r.get("score", 0) > 0.1]
        logger.info(
            "Retrieved %d relevant memories for user '%s'", len(memories), user_id
        )
    except Exception as e:
        logger.error("Memory retrieval failed: %s", e)
        memories = []

    return {"retrieved_memories": memories}


def chat_node(state: ChatState) -> dict:
    """Generate a response using the Groq LLM with injected memories."""
    memories = state.get("retrieved_memories", [])
    messages = list(state["messages"])

    # Build system prompt with memories
    system_prompt = _build_system_prompt(memories)

    # Prepend system message (replace any existing system message)
    final_messages = [SystemMessage(content=system_prompt)]
    for msg in messages:
        if not isinstance(msg, SystemMessage):
            final_messages.append(msg)

    response = llm.invoke(final_messages)
    return {"messages": [response]}


# ---------------------------------------------------------------------------
# Checkpointer
# ---------------------------------------------------------------------------
conn = sqlite3.connect(database="chatbot_memory.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# ---------------------------------------------------------------------------
# Graph:  extract → retrieve → chat
# ---------------------------------------------------------------------------
graph = StateGraph(ChatState)

graph.add_node("memory_extraction", memory_extraction_node)
graph.add_node("memory_retrieval", memory_retrieval_node)
graph.add_node("chat_node", chat_node)

graph.add_edge(START, "memory_extraction")
graph.add_edge("memory_extraction", "memory_retrieval")
graph.add_edge("memory_retrieval", "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Helpers (used by the Streamlit frontend)
# ---------------------------------------------------------------------------
def retrieve_all_threads():
    """Return all known thread IDs from the checkpoint store."""
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)
