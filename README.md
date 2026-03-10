# SmartHire AI Agent 🤝 🚀

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_svg?style=flat)](https://smarthire-agent-p33jyvkce8ojv6tygunkjk.streamlit.app/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-blue)](https://www.langchain.com/langgraph)
[![Groq](https://img.shields.io/badge/LLM-Groq-orange)](https://groq.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**SmartHire-Agent** is a professional, high-performance AI Recruitment and Intelligent Chatbot ecosystem powered by **LangGraph**. It bridges the gap between hiring managers and top talent by automating the entire recruitment lifecycle—from initial screening to automated interview scheduling—while maintaining a personal touch through long-term memory.

### 🔗 [Live Demo: smarthire-agent.streamlit.app](https://smarthire-agent-p33jyvkce8ojv6tygunkjk.streamlit.app/)

---

## 🌟 Key Features

### 1. Enterprise Recruitment Workflow
*   **Intelligent Resume Parsing**: High-accuracy extraction from PDF and DOCX files (handles encryption and complex layouts).
*   **Dynamic Requirement Matching**: Evaluates candidates against complex job descriptions with multidimensional scoring (Skills, Experience, Projects).
*   **Automated Communication**:
    *   **Zoom Integration**: On-the-fly generation of unique interview meeting links via Zoom Server-to-Server OAuth.
    *   **SMTP Email Automation**: Professional, automated email dispatch for shortlisting or polite rejections.
*   **Data-Driven Reports**: Generates comprehensive executive summaries of the recruitment pipeline.

### 2. Advanced Intelligent Chatbot
*   **Long-Term Memory Persistence**: Remembers names, preferences, and context across sessions using **PostgreSQL** and **pgvector**.
*   **Vectorless RAG (Document Q&A)**: "Chat with your files" capability using a native Python MCP (Model Context Protocol) implementation for efficient document indexing.
*   **Self-Correction & Tools**: Dynamically scales between internal knowledge, live web searches (DuckDuckGo), and arithmetic tools.

---

## 🛠️ Technical Architecture

*   **Brain**: LangGraph (Unified State Management) for sophisticated multi-node agent behavior.
*   **LLM Engine**: Groq (Llama 3.1 & 3.3) for sub-second inference speeds.
*   **UI/UX**: Streamlit with custom CSS and dynamic state handling.
*   **Memory Store**: PostgreSQL with `pgvector` for semantic search of user profiles.
*   **Doc Indexing**: Native MCP Server implementation for high-speed document processing on Streamlit Cloud.

---

## 🚀 Deployment & Installation

### Prerequisites
- Python 3.10+
- PostgreSQL (with `pgvector` extension enabled)
- Cloud API Keys: Groq, Zoom (S2S OAuth), SMTP (Gmail App Password recommended)

### Local Setup
1. **Clone the Repo**:
   ```bash
   git clone https://github.com/MahadevJagtap/SmartHire-Agent.git
   cd SmartHire-Agent
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure environment**:
   Create a `.env` file based on the template below:
   ```env
   GROQ_API_KEY=xxx
   DATABASE_URL=postgresql://user:pass@host:5432/db
   PAGEINDEX_MCP_PATH=pageindex/server.py
   EMAIL_ADDRESS=xxx@gmail.com
   EMAIL_PASSWORD=xxx (App Password)
   ZOOM_ACCOUNT_ID=xxx
   ZOOM_CLIENT_ID=xxx
   ZOOM_CLIENT_SECRET=xxx
   ```
4. **Run App**:
   ```bash
   streamlit run streamlit_app.py
   ```

---

## 📂 Project Highlights

*   `agents/`: StateGraph definitions orchestrating recruitment and chat nodes.
*   `pageindex/`: Native Python MCP server for file indexing and retrieval.
*   `services/`: Business logic for scoring, parsing, and data transformation.
*   `tools/`: Modular toolset (Zoom, Email, Search, Scoring).
*   `models/`: SQLAlchemy and Pydantic models for data integrity.

---

## 📄 License & Contributions

Licensed under the MIT License. Contributions are welcome! Please submit a PR or open an issue for feature requests.

---
*Developed with focus on speed, reliability, and user experience.*
