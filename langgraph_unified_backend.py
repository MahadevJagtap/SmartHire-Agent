"""
Unified LangGraph Backend
==========================
Combines ALL features into a single backend:
  • Tools:   DuckDuckGo search, calculator, stock price
  • MCP:     Expense tracker + PageIndex (vectorless RAG)
  • Memory:  Long-term user memory (pgvector) with auto-extraction
  • Persistence: Async SQLite checkpointer

Graph flow:
  START → memory_extraction → memory_retrieval → chat_node ⇄ tools → END
"""

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import (
    BaseMessage, HumanMessage, SystemMessage, AIMessage,
)
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool, BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
from memory_manager import (
    get_relevant_memories,
    auto_store_memories,
    get_user_memories,
)
import aiosqlite
import requests
import asyncio
import threading
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
# Dedicated async event loop (Streamlit compatibility)
# ---------------------------------------------------------------------------
_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()


def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)


def run_async(coro):
    return _submit_async(coro).result()


def submit_async_task(coro):
    """Schedule a coroutine on the backend event loop."""
    return _submit_async(coro)


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
llm = ChatGroq(model="llama-3.1-8b-instant")

# ---------------------------------------------------------------------------
# Tools — native
# ---------------------------------------------------------------------------
search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        ops = {
            "add": lambda a, b: a + b,
            "sub": lambda a, b: a - b,
            "mul": lambda a, b: a * b,
            "div": lambda a, b: a / b if b != 0 else "Division by zero",
        }
        if operation not in ops:
            return {"error": f"Unsupported operation '{operation}'"}
        result = ops[operation](first_num, second_num)
        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')
    using Alpha Vantage.
    """
    url = (
        f"https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}"
        f"&apikey=C9PE94QUEW9VWGFM"
    )
    r = requests.get(url)
    return r.json()


# ---------------------------------------------------------------------------
# Tools — MCP (expense + PageIndex)
# ---------------------------------------------------------------------------
# This allows Streamlit Cloud to read the path from Secrets via environment variables.
PAGEINDEX_PATH = os.getenv("PAGEINDEX_MCP_PATH", "pageindex/server.py")

# Determine command based on file extension
mcp_command = "python" if PAGEINDEX_PATH.endswith(".py") else "node"

client = MultiServerMCPClient(
    {
        "expense": {
            "transport": "streamable_http",
            "url": "https://splendid-gold-dingo.fastmcp.app/mcp",
        },
        "pageindex": {
            "transport": "stdio",
            "command": mcp_command,
            "args": [PAGEINDEX_PATH],
        },
    }
)


def load_mcp_tools() -> list[BaseTool]:
    try:
        return run_async(client.get_tools())
    except Exception as e:
        logger.warning("Could not load MCP tools: %s", e)
        return []


mcp_tools = load_mcp_tools()

# Combine all tools
all_tools = [search_tool, calculator, get_stock_price, *mcp_tools]
llm_with_tools = llm.bind_tools(all_tools) if all_tools else llm


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str
    retrieved_memories: list[str]


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_TEMPLATE = """You are a powerful AI assistant with the following capabilities:
• Long-term memory — you remember important user facts across sessions
• Web search — you can search the internet for current information
• Calculator — you can perform arithmetic operations
• Stock prices — you can fetch live stock data
• Document Q&A — you can index and search uploaded documents via PageIndex
• Expense tracking — you can manage expenses via the expense tool

Use stored memories to personalize responses naturally.
Do NOT tell the user you are "storing" or "retrieving" memories.

{memories_section}"""


def _build_system_prompt(memories: list[str]) -> str:
    if memories:
        bullets = "\n".join(f"• {m}" for m in memories)
        section = f"User memories:\n{bullets}"
    else:
        section = "No stored memories for this user yet."
    return SYSTEM_TEMPLATE.format(memories_section=section)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def memory_extraction_node(state: ChatState) -> dict:
    """Extract personal facts from the latest user message and store them."""
    user_id = state.get("user_id", "default_user")
    latest = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            latest = msg.content
            break
    if not latest:
        return {}
    try:
        stored = auto_store_memories(user_id, latest)
        if stored:
            logger.info("Auto-extracted %d memories for '%s'", len(stored), user_id)
    except Exception as e:
        logger.error("Memory extraction failed: %s", e)
    return {}


def memory_retrieval_node(state: ChatState) -> dict:
    """Retrieve relevant memories for the current query."""
    user_id = state.get("user_id", "default_user")
    latest = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            latest = msg.content
            break
    if not latest:
        return {"retrieved_memories": []}
    try:
        results = get_relevant_memories(user_id, latest, k=5)
        memories = [r["content"] for r in results if r.get("score", 0) > 0.1]
    except Exception as e:
        logger.error("Memory retrieval failed: %s", e)
        memories = []
    return {"retrieved_memories": memories}


async def chat_node(state: ChatState) -> dict:
    """Generate a response with memory-augmented prompt and tool access."""
    memories = state.get("retrieved_memories", [])
    messages = list(state["messages"])

    system_prompt = _build_system_prompt(memories)
    final_messages = [SystemMessage(content=system_prompt)]
    for msg in messages:
        if not isinstance(msg, SystemMessage):
            final_messages.append(msg)

    response = await llm_with_tools.ainvoke(final_messages)
    return {"messages": [response]}


# ---------------------------------------------------------------------------
# Checkpointer (async SQLite)
# ---------------------------------------------------------------------------
async def _init_checkpointer():
    conn = await aiosqlite.connect(database="chatbot_unified.db")
    return AsyncSqliteSaver(conn)


checkpointer = run_async(_init_checkpointer())

# ---------------------------------------------------------------------------
# Graph: extract → retrieve → chat ⇄ tools
# ---------------------------------------------------------------------------
graph = StateGraph(ChatState)

graph.add_node("memory_extraction", memory_extraction_node)
graph.add_node("memory_retrieval", memory_retrieval_node)
graph.add_node("chat_node", chat_node)

graph.add_edge(START, "memory_extraction")
graph.add_edge("memory_extraction", "memory_retrieval")
graph.add_edge("memory_retrieval", "chat_node")

tool_node = ToolNode(all_tools) if all_tools else None
if tool_node:
    graph.add_node("tools", tool_node)
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")
else:
    graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
async def _alist_threads():
    all_threads = set()
    async for checkpoint in checkpointer.alist(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


def retrieve_all_threads():
    return run_async(_alist_threads())
