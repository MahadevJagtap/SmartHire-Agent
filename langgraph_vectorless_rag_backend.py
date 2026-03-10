from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool, BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
import aiosqlite
import asyncio
import threading
import os

load_dotenv()

# Dedicated async loop for backend tasks (Streamlit compatibility pattern)
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

# -------------------
# 1. LLM
# -------------------
llm = ChatGroq(model="llama-3.3-70b-versatile")

# -------------------
# 2. Tools
# -------------------
search_tool = DuckDuckGoSearchRun(region="us-en")

# ⭐ PageIndex Vectorless RAG MCP Client
PAGEINDEX_PATH = os.getenv("PAGEINDEX_MCP_PATH", r"C:\Users\Shubh\OneDrive\Desktop\pageindex-mcp\build\index.js")

client = MultiServerMCPClient(
    {
        "pageindex": {
            "transport": "stdio",
            "command": "node",
            "args": [PAGEINDEX_PATH],
        }
    }
)

def load_mcp_tools() -> list[BaseTool]:
    try:
        return run_async(client.get_tools())
    except Exception as e:
        print(f"Error loading MCP tools: {e}")
        return []

mcp_tools = load_mcp_tools()
tools = [search_tool] + mcp_tools
llm_with_tools = llm.bind_tools(tools) if tools else llm

# -------------------
# 3. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# -------------------
# 4. Nodes
# -------------------
async def chat_node(state: ChatState, config=None):
    """LLM node that handles conversation and tool calling."""
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    
    system_prompt = SystemMessage(
        content=(
            "You are a Vectorless RAG Assistant. Use the `pageindex` tools for searching information "
            "in indexed documents and for indexing new content. Do NOT use local embeddings or vector stores. "
            f"Active Thread ID: {thread_id}"
        )
    )
    
    messages = [system_prompt] + state["messages"]
    response = await llm_with_tools.ainvoke(messages, config=config)
    return {"messages": [response]}

tool_node = ToolNode(tools) if tools else None

# -------------------
# 5. Checkpointer
# -------------------
async def _init_checkpointer():
    conn = await aiosqlite.connect(database="chatbot_rag.db")
    return AsyncSqliteSaver(conn)

checkpointer = run_async(_init_checkpointer())

# -------------------
# 6. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")

if tool_node:
    graph.add_node("tools", tool_node)
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")
else:
    graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

# -------------------
# 7. Helpers
# -------------------
async def _alist_threads():
    all_threads = set()
    async for checkpoint in checkpointer.alist(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)

def retrieve_all_threads():
    return run_async(_alist_threads())
