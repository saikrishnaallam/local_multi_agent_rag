# Local Multi-Agent RAG System

An autonomous multi-agent routing system built using **LangGraph**, **LangChain**, and a local **Llama 3** model running via **Ollama**.

The system dynamically routes questions between:
1. **Live Web Search Agent**: Queries DuckDuckGo for current events/general knowledge.
2. **Local RAG Agent**: Queries a local, in-memory vector database (ChromaDB) for document-specific questions.

## 📊 System Architecture Flowchart

```mermaid
flowchart TD
    Start([User Question]) --> Router[🧠 Supervisor Router]
    Router -->|search| WebSearch[🌐 Web Search Agent]
    Router -->|rag| LocalRAG[📄 Local RAG Agent]
    
    WebSearch --> DDG[🔍 DuckDuckGo API]
    DDG --> WebSynth[🤖 Llama3 Synthesis]
    
    LocalRAG --> Chroma[📦 Chroma DB Lookup]
    Chroma --> RAGSynth[🤖 Llama3/Chroma Context Synthesis]
    
    WebSynth --> End([Final Response])
    RAGSynth --> End
    
    classDef agent fill:#f9f,stroke:#333,stroke-width:2px;
    classDef tool fill:#bbf,stroke:#333,stroke-width:1px;
    classDef io fill:#bfb,stroke:#333,stroke-width:1px;
    
    class Router,WebSearch,LocalRAG agent;
    class DDG,Chroma tool;
    class Start,End io;
```

---

## 🛠️ Step-by-Step Setup Procedure

### 1. Install and Start Ollama
1. Download **Ollama** for macOS/Windows/Linux from [ollama.com](https://ollama.com).
2. Install and launch the application. Ensure the Ollama icon is visible in your menu bar (macOS) or system tray.
3. Open your terminal and download the **Llama 3** model:
   ```bash
   ollama pull llama3
   ```
4. Verify Ollama is running and has the model installed:
   ```bash
   ollama list
   ```

> [!TIP]
> You can also run other local models (e.g., `llama3.1`, `llama3.2`, or `mistral`) by pulling them via Ollama (`ollama pull <model-name>`) and updating the model parameter in `sanity_check.py` and `agent_system.py`.

### 2. Configure the Python Virtual Environment
Clone the repository, navigate into the directory, and set up your virtual environment:

```bash
# Clone the repository
git clone https://github.com/saikrishnaallam/local_multi_agent_rag.git

# Navigate to project directory
cd local_multi_agent_rag

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

### 1. Run the Connection Sanity Check
Before launching the agent workflow, run the sanity check to confirm your local model is accessible via Ollama:

```bash
python sanity_check.py
```

**Expected Output:**
```text
🔄 Connecting to local Llama3 model via Ollama...

✅ Success! Your local AI brain is online.
🤖 AI Response: [Confirmation response from Llama3]
```

### 2. Run the Multi-Agent System
Run the main system to execute test cases for both the web search and local RAG agents:

```bash
python agent_system.py
```

**Expected Output:**
```text
=============================================
🚀 RUNNING FULL LANGGRAPH MULTI-AGENT SYSTEM
=============================================

--- 📝 Test Case 1: Requesting Live Web Data ---
🧠 Supervisor is analyzing the request...
🎯 Router Decision: Directed to -> search
🌐 Web Search Agent is activating...
🔍 Searching the live web for: 'Who won the latest Super Bowl and what was the score?'...
🤖 Final Graph Response (Web Search):
Based on the search results, the latest Super Bowl (Super Bowl LVIII) was won by the Kansas City Chiefs, who defeated the San Francisco 49ers with a final score of 25-22 on February 11, 2024.

--- 📝 Test Case 2: Requesting Document Data ---
🧠 Supervisor is analyzing the request...
🎯 Router Decision: Directed to -> rag
📄 Local RAG Agent is activating...
🔎 Querying local vector database for: 'Based on the uploaded project document, what database engine does Project Alpha use?'...
🤖 Local model is generating a response based on document data...

🤖 Final Graph Response (Local RAG):
Based on the provided document context, Project Alpha uses PostgreSQL version 15 as its primary database engine.
```

---

## 🧠 Routing Behavior (Search vs RAG)

The **Supervisor Router** uses a prompt-based classification logic to decide where to send queries:
- **Web Search (`search`)**: Triggered for general knowledge, current events, or live web info (e.g., *"Who won the latest Super Bowl and what was the score?"*).
- **Document RAG (`rag`)**: To trigger the RAG agent, the prompt instructions require the query to explicitly refer to **uploaded files, documents, essays, or notes** (e.g., *"Based on the uploaded project overview document, what database engine does Project Alpha use?"*). 

> [!NOTE]
> If a query is generic (e.g., *"What database engine does Project Alpha use?"*), the local router will likely classify it as **search** because it is framed as general knowledge rather than a document-specific query.

---

## 🔍 Troubleshooting & Common Issues

### 1. Connection Failed Error
* **Error**: `❌ Connection Failed. Error Details: ...`
* **Fix**: Ensure the Ollama app is open and running in your Mac's menu bar or system tray.

### 2. Model Not Found (404)
* **Error**: `model 'llama3' not found (status code: 404)`
* **Fix**: Run `ollama pull llama3` in your terminal to download the model weights locally.

### 3. Missing Integration / Import Errors
* **Error**: `ImportError: cannot import name 'ChatOllama' from 'langchain_community.chat_models'`
* **Fix**: Make sure you have installed `langchain-ollama` and are importing via `from langchain_ollama import ChatOllama`.
* **Error**: `ImportError: Could not import chromadb`
* **Fix**: Run `pip install chromadb` inside your active virtual environment.
* **Error**: `ImportError: Could not import duckduckgo_search`
* **Fix**: Run `pip install duckduckgo-search` inside your active virtual environment.

---

## 🏗️ Architecture Under the Hood
- **StateGraph**: LangGraph manages the shared agent state (`messages` and `next_step`).
- **Supervisor Router**: Classifies the query (`search` or `rag`) using Llama 3.
- **Web Search Node**: Activates the DuckDuckGo Search tool to fetch raw internet results.
- **Local RAG Node**: Splits and embeds a mock project overview document (`project_alpha_guide.pdf`) using `OllamaEmbeddings`, indexes it into Chroma, retrieves context, and answers the query.

---

## 📂 Project Structure

- [agent_system.py](file:///Users/saikrishnaallam/Desktop/local_multi_agent_rag/agent_system.py): The main multi-agent implementation using LangGraph, including the Supervisor Router, Web Search Agent, and Local RAG Agent nodes.
- [sanity_check.py](file:///Users/saikrishnaallam/Desktop/local_multi_agent_rag/sanity_check.py): A quick validation script to verify local connectivity to Ollama and check if the Llama 3 model is running.
- [requirements.txt](file:///Users/saikrishnaallam/Desktop/local_multi_agent_rag/requirements.txt): Defines Python package dependencies required to run the agents.
