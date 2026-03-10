# SmartHire AI Agent 🚀

**SmartHire-Agent** is an advanced, LangGraph-powered AI recruitment and chatbot ecosystem. It streamlines the hiring process by automating resume screening, candidate communication, and interview scheduling, while providing a persistent memory-enabled chat interface.

---

## 🌟 Core Features

### 1. AI Recruitment Agent
- **Automated Screening**: Analyzes resumes against job descriptions using state-of-the-art LLMs.
- **Smart Shortlisting**: Scores candidates and provides detailed reasoning for selection or rejection.
- **Interview Automation**: 
    - Generates real-time **Zoom Meeting** links (with fallback to Personal Meeting Rooms).
    - Sends professional **SMTP-based emails** to candidates with interview details.
- **Unified Dashboard**: A sleek Streamlit interface to manage the entire recruitment pipeline.

### 2. Intelligent Chatbot Ecosystem
- **Long-Term Memory**: Remembers user preferences and past interactions using a PostgreSQL/pgvector backend.
- **Vectorless RAG**: Efficiently indexes and retrieves information from uploaded documents (PDFs, etc.) without complex vector database overhead.
- **Tool Integration**: Capable of using external tools (web search, calculators, etc.) dynamically.
- **Multi-Modal**: Handles document uploads and complex reasoning tasks.

---

## 🛠️ Tech Stack

- **Framework**: [LangGraph](https://www.langchain.com/langgraph) (Unified Orchestration)
- **Frontend**: [Streamlit](https://streamlit.io/) (Multi-page Interactive UI)
- **LLM**: [Groq](https://groq.com/) / LangChain
- **Database**: PostgreSQL with `pgvector` (Long-term Memory)
- **Integrations**: 
    - **Zoom API** (Server-to-Server OAuth)
    - **SMTP** for automated Emailing

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- PostgreSQL (for memory persistence)
- Groq API Key
- Zoom Server-to-Server OAuth Credentials

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/MahadevJagtap/SmartHire-Agent.git
   cd SmartHire-Agent
   ```

2. **Set up Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**:
   Create a `.env` file in the root directory and add your credentials:
   ```env
   # LLM & Memory
   GROQ_API_KEY=your_key
   DATABASE_URL=postgresql://user:password@localhost:5432/db

   # Email (SMTP)
   EMAIL_ADDRESS=your_email@gmail.com
   EMAIL_PASSWORD=your_app_password
   SMTP_SERVER=smtp.gmail.com
   SMTP_PORT=587

   # Zoom API
   ZOOM_ACCOUNT_ID=...
   ZOOM_CLIENT_ID=...
   ZOOM_CLIENT_SECRET=...
   PERSONAL_ZOOM_LINK=...
   ```

### Running the Application

Start the unified Streamlit interface:
```bash
streamlit run streamlit_app.py
```

---

## 📂 Project Structure

- `agents/`: Contains specialized LangGraph agent definitions (Recruitment, Chatbot).
- `services/`: Core logic for recruitment scoring and candidate processing.
- `tools/`: Custom tools for Zoom link generation and Email dispatch.
- `models/`: Database schemas and data models.
- `streamlit_app.py`: Main entry point for the unified UI.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---
*Created with ❤️ by the SmartHire Team.*
