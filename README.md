# Local Multi-Agent Corrective RAG (CRAG) System 🤖

🚀 **A 100% private, local, and autonomous multi-agent Corrective RAG (CRAG) system.** Built using **LangGraph**, **LangChain**, and **Streamlit**, powered by local models running via **Ollama**.

---

### 🌟 Project Status & Badges
[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.60.0-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangChain Framework](https://img.shields.io/badge/LangChain-0.3-orange.svg?logo=chainlink&logoColor=white)](https://www.langchain.com/)
[![LangGraph Orchestrator](https://img.shields.io/badge/LangGraph-Active-darkgreen.svg)](https://github.com/langchain-ai/langgraph)
[![Ollama Engine](https://img.shields.io/badge/Ollama-Llama_3_%26_Nomic-black.svg)](https://ollama.com/)

---

## ✨ Key Highlights

*   **🛡️ 100% Local & Private**: No data leaves your machine. Generative reasoning (Llama 3) and text representations (`nomic-embed-text`) execute fully locally.
*   **⚖️ Self-Corrective (CRAG) Routing**: The agent does not blindly trust database search. It retrieves context, grades it using a strict Pydantic model, and automatically falls back to **DuckDuckGo Web Search** if your files don't have the answer.
*   **📄 Smarter Multi-Document Indexing**: Upload multiple PDF documents at once. The retriever extracts text chunks, prepends source document metadata tags, and feeds context from all files to the LLM in a single turn.
*   **🕵️‍♂️ Real-Time Agent Trace**: Watch the LangGraph state machine execute nodes (`retrieve` ➡️ `web_search` ➡️ `generate`) inside an expandable visual status console.
*   **💾 SQLite Session Synchronization**: Fully integrated Streamlit memory configuration guarantees that your query retriever and document indexer share the same database connections across browser reload cycles.

---

## 📊 System Architecture & Data Flow

Our agent runs on a compiled state machine, ensuring structured execution paths based on real-time grading:

```mermaid
flowchart TD
    Start([User Query]) -->|Query Vector Store| Retrieve[🔍 retrieve Node]
    Retrieve -->|Grade Context| Grade[⚖️ grade_documents_node]
    
    Grade -->|Relevance Match / generate| Generate[🤖 generate Node]
    Grade -->|Irrelevant Chunks / web_search| WebSearch[🌐 web_search Node]
    
    WebSearch -->|Web Context| Generate
    Generate --> End([Synthesized Response])

    classDef nodeStyle fill:#4A154B,stroke:#333,stroke-width:2px,color:#fff;
    classDef ioStyle fill:#007A5E,stroke:#333,stroke-width:1px,color:#fff;
    class Retrieve,Grade,Generate,WebSearch nodeStyle;
    class Start,End ioStyle;
```

---

## 🔄 Previous vs. Current Architecture

We did extensive refactoring to turn a CLI script into a fully operational local RAG workspace.

| Feature | Legacy Setup (`agent_system.py`) | Refactored Setup (`main.py` + `app.py`) | Why we changed it |
| :--- | :--- | :--- | :--- |
| **User Interface** | 💻 Terminal console executing hardcoded queries. | 🌐 **Web Chat UI** with sidebars, upload drag-and-drop slots, and node tracking expanders. | To make the local workspace accessible, user-friendly, and interactive. |
| **Vector DB Lifecycle** | 🧠 In-Memory Chroma DB (seeded fresh on startup). | 📦 **Persistent Local Chroma DB** (`./chroma_db`) with file caching. | Saves processing time by storing indexed documents locally on disk. |
| **Routing Accuracy** | 🧠 Supervisor router classified query paths before vector retrieval. | ⚖️ **Corrective RAG (CRAG)**: Retrieves data first, then uses Pydantic structured grading. | Prevents hallucinations by validating data relevance before generation. |
| **Session Memory** | ❌ None (lost when terminal process exits). | 💾 **Streamlit `st.session_state` Connection Binding**. | Prevents Streamlit's script-reload cycles from creating conflicting database descriptors. |
| **Multiple File Support** | ❌ Restricted to `k=3` chunks (crowds out extra PDFs). | ✅ **Expanded `k=10` with custom Source Metadata Tagging**. | Supports cross-document summarization by mapping chunk boundaries directly to source files. |

---

## 🧠 Under the Hood: Key Design Decisions

> [!IMPORTANT]
> **No Disk-Level Folder Deletes (`shutil.rmtree`)**
> Deleting database directories while an active process holds SQLite connections throws `readonly database` errors. We use the native Chroma SQL-level API `get()["ids"]` and `delete(ids)` to wipe files cleanly without file lock exceptions.

> [!TIP]
> **Strict Grader Tool Binding**
> Prompting local models to output simple strings often leads to unexpected parsing errors. We wrap the grader inside a Pydantic model (`RouteDecision`) and enforce structured JSON schemas via Ollama's tool binding.

---

## 🛠️ Step-by-Step Quick Start

### 1. Install and Start Ollama
1. Download **Ollama** for macOS/Windows/Linux from [ollama.com](https://ollama.com).
2. Install and launch the application.
3. Open your terminal and pull the models:
   ```bash
   # Llama 3 for reasoning, routing and synthesis
   ollama pull llama3
   
   # Nomic embeddings for document representations
   ollama pull nomic-embed-text
   ```

### 2. Configure Python Virtual Environment
*(Recommended Python Version: 3.9, 3.10, or 3.11)*

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Running the Project

### 1. Verify Local Model Connection (Recommended)
Before launching the applications, run the sanity check to confirm your local models are correctly running and accessible via Ollama:
```bash
python sanity_check.py
```

### 2. Run the Web Interface (Recommended)
Launch the interactive web assistant:
```bash
streamlit run app.py
```
* Access the app in your browser at `http://localhost:8501`.
* Drag and drop multiple PDF files in the sidebar, click **Process Documents**, and start chatting.

### 3. Run CLI Ingestion (Alternative)
You can also index documents directly using the terminal utility:
```bash
python ingest.py path/to/document.pdf
```

### 4. Run Corrective RAG CLI (Alternative)
You can test the main Corrective RAG (CRAG) state machine directly in the terminal:
```bash
python main.py
```
This script runs preconfigured test cases:
* **In-Context Query**: Asks about "Project Alpha" (expects local database retrieval ➡️ answer generation).
* **Out-of-Context Query**: Asks about the "Super Bowl" (expects retrieval ➡️ fallback to DuckDuckGo Web Search ➡️ answer generation).

### 5. Run Legacy Terminal Agent System (Optional)
If you want to run the original supervisor-routed hybrid search agent with Cross-Encoder reranking in your terminal:
```bash
python agent_system.py
```

---

## 🛠️ Visualizing & Debugging with LangSmith (Optional)

Since this system is built on LangGraph and LangChain, you can easily trace the agent steps, document retrieves, grader decisions, and generative prompts in a beautiful visual UI using **LangSmith**.

To enable tracing:
1. Sign up for a free account at [smith.langchain.com](https://smith.langchain.com).
2. Generate an API key from your profile settings.
3. Export the following environment variables in your terminal before running the python applications:
   ```bash
   export LANGCHAIN_TRACING_V2="true"
   export LANGCHAIN_API_KEY="your-api-key-here"
   export LANGCHAIN_PROJECT="local-multi-agent-rag"
   ```
4. Run the scripts or Streamlit interface as normal. All node transitions and model calls will automatically be captured and visualized in your LangSmith dashboard!

---

## 📂 Project Structure

*   [app.py](app.py): Streamlit web application managing uploader actions, memory states, and real-time execution steps.
*   [main.py](main.py): Corrective RAG (CRAG) graph workflow connecting structured document grading and fallback search.
*   [ingest.py](ingest.py): CLI loading utility which splits and indexes PDFs into `./chroma_db`.
*   [agent_system.py](agent_system.py): Legacy supervisor-routed in-memory RAG implementation featuring Hybrid Search (Vector + BM25) and Cross-Encoder reranking, populated with mock DB seeds.
*   [sanity_check.py](sanity_check.py): Verifies local connection to Ollama and loaded model states.
*   [project_alpha_overview.pdf](project_alpha_overview.pdf): Sample PDF document containing confidential overview details for Project Alpha (used for testing vector ingestion).
*   [requirements.txt](requirements.txt): Python dependency configurations.
*   [.gitignore](.gitignore): Excludes cache directories and local document files.
