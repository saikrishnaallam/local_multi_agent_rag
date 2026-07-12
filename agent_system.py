# Add this import at the very top with your other imports
from langgraph.graph import StateGraph, START, END

from typing import TypedDict, Annotated, Literal
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.tools import DuckDuckGoSearchRun

# 🌟 NEW IMPORTS FOR STEP 4
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# 1. Define the Shared Memory (State)
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    next_step: str

# 2. Initialize the Local Brain, Search Tool, and Local Embeddings
local_llm = ChatOllama(model="llama3", temperature=0)
web_search_tool = DuckDuckGoSearchRun()
# We use the same local llama3 model to generate our mathematical vector embeddings!
local_embeddings = OllamaEmbeddings(model="llama3")

# Global placeholder for our vector database
vector_db = None

# --- Existing Router Node (from Step 2) ---
def supervisor_router(state: AgentState) -> Literal["rag", "search"]:
    print("\n🧠 Supervisor is analyzing the request...")
    last_message = state["messages"][-1].content
    system_prompt = SystemMessage(
        content=(
            "You are an expert routing assistant for a multi-agent system.\n"
            "Analyze the user's query and determine the best source:\n"
            "- Reply ONLY with the word 'rag' if the user is asking about uploaded files, documents, essays, or notes.\n"
            "- Reply ONLY with the word 'search' if the user is asking about current events, live web information, code generation, or general knowledge.\n"
            "Do not include punctuation, spaces, or any other words. Output exactly 'rag' or 'search'."
        )
    )
    user_message = HumanMessage(content=last_message)
    decision = local_llm.invoke([system_prompt, user_message]).content.strip().lower()
    print(f"🎯 Router Decision: Directed to -> {decision}")
    return decision

# --- Existing Web Search Node (from Step 3) ---
def web_search_agent(state: AgentState):
    print("\n🌐 Web Search Agent is activating...")
    user_query = state["messages"][-1].content
    print(f"🔍 Searching the live web for: '{user_query}'...")
    try:
        search_results = web_search_tool.run(user_query)
    except Exception as e:
        search_results = f"Search failed or timed out. Error: {e}"
    
    system_message = SystemMessage(
        content="You are an autonomous Research Agent. Write a factual response based strictly on the search results provided."
    )
    combined_input = HumanMessage(content=f"User Question: {user_query}\n\nRaw Internet Search Results:\n{search_results}")
    ai_response = local_llm.invoke([system_message, combined_input])
    return {"messages": [AIMessage(content=ai_response.content)]}


# 🌟 NEW NODE: Local RAG Agent Node Logic
def rag_agent(state: AgentState):
    """Queries the local vector database and generates an answer strictly using document context."""
    print("\n📄 Local RAG Agent is activating...")
    global vector_db
    
    # 1. Extract the user query
    user_query = state["messages"][-1].content
    print(f"🔎 Querying local vector database for: '{user_query}'...")
    
    if vector_db is None:
        return {"messages": [AIMessage(content="Error: No documents have been indexed into the vector database yet.")]}
    
    # 2. Retrieve top 3 most relevant document chunks
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(user_query)
    
    # Combine the text chunks into a single context string
    retrieved_context = "\n\n".join([d.page_content for d in docs])
    
    # 3. Formulate the system prompt to prevent hallucinations
    system_message = SystemMessage(
        content=(
            "You are a strict study companion assistant. Answer the user's question using ONLY the provided "
            "document context below. If the answer cannot be found in the context, say exactly 'I cannot find that in the documents.' "
            "Do not make things up."
        )
    )
    
    combined_input = HumanMessage(
        content=f"User Question: {user_query}\n\nRetrieved Document Context:\n{retrieved_context}"
    )
    
    # 4. Generate the response using our local Llama3 model
    print("🤖 Local model is generating a response based on document data...")
    ai_response = local_llm.invoke([system_message, combined_input])
    
    return {"messages": [AIMessage(content=ai_response.content)]}


# --- Helper Function to Seed Fake Data for Testing ---
def seed_mock_vector_db():
    """Simulates uploading a document so our RAG agent has something to read."""
    global vector_db
    print("📦 Seeding local vector database with mock textbook data...")
    
    # Sample document text simulating a study guide
    mock_document_text = (
        "Project Alpha is a confidential software initiative running on a local cluster. "
        "The primary database engine utilized is PostgreSQL version 15. The system uses "
        "Redis for caching web sessions, ensuring a response latency under 15 milliseconds. "
        "The project lead is Sarah Jenkins, and it is scheduled for deployment in Q4 2026."
    )
    
    # Wrap text in a LangChain Document structure
    doc = Document(page_content=mock_document_text, metadata={"source": "project_alpha_guide.pdf"})
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    chunks = text_splitter.split_documents([doc])
    
    # Build a local, in-memory Chroma instance using local Ollama embeddings
    vector_db = Chroma.from_documents(chunks, local_embeddings)
    print("✅ Mock Vector DB successfully initialized and loaded into memory.")

# ... (Keep all your existing nodes: supervisor_router, web_search_agent, rag_agent) ...

# 1. Initialize the StateGraph with our custom AgentState
workflow = StateGraph(AgentState)

# 2. Add the execution nodes to the graph
workflow.add_node("web_search", web_search_agent)
workflow.add_node("local_rag", rag_agent)

# 3. Set up the conditional routing logic from the starting line
workflow.add_conditional_edges(
    START,                # The default starting point of the graph
    supervisor_router,    # The function that decides where to go
    {
        "search": "web_search",
        "rag": "local_rag"
    }
)

# 4. Connect the output of both agents to the END node
workflow.add_edge("web_search", END)
workflow.add_edge("local_rag", END)

# 5. Compile the graph into a single executable application
local_agent_app = workflow.compile()

# --- Upgraded Local Graph Verification Test ---
if __name__ == "__main__":
    # 1. Initialize our mock database for the RAG agent
    seed_mock_vector_db()
    
    print("\n=============================================")
    print("🚀 RUNNING FULL LANGGRAPH MULTI-AGENT SYSTEM")
    print("=============================================\n")
    
    # --- TEST CASE 1: Testing the Web Search Branch ---
    print("--- 📝 Test Case 1: Requesting Live Web Data ---")
    query_1 = "Who won the latest Super Bowl and what was the score?"
    inputs_1 = {"messages": [HumanMessage(content=query_1)]}
    
    # Run the compiled LangGraph application!
    output_1 = local_agent_app.invoke(inputs_1)
    
    print("\n🤖 Final Graph Response (Web Search):")
    print(output_1["messages"][-1].content)
    print("\n---------------------------------------------\n")
    
    # --- TEST CASE 2: Testing the Local RAG Branch ---
    print("--- 📝 Test Case 2: Requesting Document Data ---")
    query_2 = "Based on the uploaded project document, what database engine does Project Alpha use?"
    inputs_2 = {"messages": [HumanMessage(content=query_2)]}
    
    # Run the compiled LangGraph application again!
    output_2 = local_agent_app.invoke(inputs_2)
    
    print("\n🤖 Final Graph Response (Local RAG):")
    print(output_2["messages"][-1].content)
    print("\n=============================================")