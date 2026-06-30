# Local Multi-Agent RAG System

An autonomous multi-agent routing system built using **LangGraph**, **LangChain**, and a local **Llama 3** model running via **Ollama**.

The system dynamically routes questions between:
1. **Live Web Search Agent**: Queries DuckDuckGo for current events/general knowledge.
2. **Local RAG Agent**: Queries a local, in-memory vector database (ChromaDB) for document-specific questions.

---

## 🛠️ Step-by-Step Setup Procedure

### 1. Download and Start Ollama
1. Download **Ollama** for macOS/Windows/Linux from [ollama.com](https://ollama.com).
2. Install and launch the application.
3. Open your terminal and download the **Llama 3** model:
   ```bash
   ollama pull llama3
   ```
4. Verify Ollama is running and has the model installed:
   ```bash
   ollama list
   ```

### 2. Configure the Python Virtual Environment
Clone the repository, navigate into the directory, and set up your virtual environment:

```bash
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

### 2. Run the Multi-Agent System
Run the main system to execute test cases for both the web search and local RAG agents:

```bash
python agent_system.py
```

### How it Works under the Hood:
- **StateGraph**: LangGraph manages the shared agent state (`messages` and `next_step`).
- **Supervisor Router**: Classifies the query (`search` or `rag`) using Llama 3.
- **Web Search Node**: Activates the DuckDuckGo Search tool to fetch raw internet results.
- **Local RAG Node**: Splits and embeds a mock study guide using `OllamaEmbeddings`, indexes it into Chroma, retrieves context, and answers the query.
