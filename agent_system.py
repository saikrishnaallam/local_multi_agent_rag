# Add this import at the very top with your other imports
from langgraph.graph import StateGraph, START, END

from typing import Annotated, Literal, Sequence, List
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun

# 🌟 NEW IMPORTS FOR STEP 4
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from langchain_community.retrievers import BM25Retriever

# ==========================================
# STEP 2: Define Agent State
# ==========================================
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    context: List[str]  # Explicitly stores retrieved chunks

# 2. Initialize the Local Brain, Search Tool, and Local Embeddings
local_llm = ChatOllama(model="llama3", temperature=0)
web_search_tool = DuckDuckGoSearchRun()
# We use the same local llama3 model to generate our mathematical vector embeddings!
local_embeddings = OllamaEmbeddings(model="llama3")

# Global placeholder for our vector database
vector_db = None
bm25_retriever = None

# Initialize lightweight Cross-Encoder model for reranking
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

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
def rag_agent_node(state: AgentState):
    """RAG Node with Hybrid Search (Vector + BM25) and Cross-Encoder Reranking."""
    print("\n📄 Local RAG Agent is activating...")
    global vector_db, bm25_retriever
    
    # 1. Extract the user query
    user_query = state["messages"][-1].content
    print(f"🔎 Querying local hybrid database for: '{user_query}'...")
    
    if vector_db is None or bm25_retriever is None:
        return {"messages": [AIMessage(content="Error: No documents have been indexed into the vector database yet.")]}
    
    # Step A: Hybrid Search Candidate Retrieval 🔍
    vector_docs = vector_db.similarity_search(user_query, k=5)
    bm25_docs = bm25_retriever.invoke(user_query)  # Top 5 keyword matches
    
    # Step B: Deduplicate Candidates
    combined_docs = list(
        {doc.page_content: doc for doc in vector_docs + bm25_docs}.values()
    )
    
    # Step C: Cross-Encoder Reranking ⚖️
    # Prepare query-document pairs for the Cross-Encoder
    pairs = [[user_query, doc.page_content] for doc in combined_docs]
    
    # Predict relevance scores
    scores = reranker.predict(pairs)
    
    # Pair docs with scores and sort in descending order
    scored_docs = sorted(
        zip(combined_docs, scores), key=lambda x: x[1], reverse=True
    )
    
    # Take the top 3 highest-scoring documents
    top_docs = [doc for doc, score in scored_docs[:3]]
    
    # Step D: Format Context & Prompt LLM 🤖
    retrieved_context = "\n\n".join([d.page_content for d in top_docs])
    
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
    print("🤖 Local model is generating a response based on document data with reranked context...")
    ai_response = local_llm.invoke([system_message, combined_input])
    
    return {
        "messages": [AIMessage(content=ai_response.content)],
        "context": [d.page_content for d in top_docs]
    }


# --- Helper Function to Seed Fake Data for Testing ---
def seed_mock_vector_db():
    """Simulates uploading a document so our RAG agent has something to read."""
    global vector_db, bm25_retriever
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
    
    # Build BM25 keyword retriever
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 5
    
    print("✅ Mock Vector DB and BM25 Retriever successfully initialized and loaded into memory.")

# ... (Keep all your existing nodes: supervisor_router, web_search_agent, rag_agent) ...

# ==========================================
# STEP 3: Assemble the Workflow
# ==========================================
# 1. Initialize the graph with our state
workflow = StateGraph(AgentState)

# 2. Add our operational nodes (ensure these functions are defined above this block)
workflow.add_node("web_search", web_search_agent) 
workflow.add_node("local_rag", rag_agent_node)    

# 3. Add the Supervisor Router at the starting line
workflow.add_conditional_edges(
    START,
    supervisor_router,
    {
        "search": "web_search",
        "rag": "local_rag"
    }
)

# 4. Connect both agent endpoints to the finish line
workflow.add_edge("web_search", END)
workflow.add_edge("local_rag", END)

# 5. Compile the executable application
agent_app = workflow.compile()

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
    output_1 = agent_app.invoke(inputs_1)
    
    print("\n🤖 Final Graph Response (Web Search):")
    print(output_1["messages"][-1].content)
    print("\n---------------------------------------------\n")
    
    # --- TEST CASE 2: Testing the Local RAG Branch ---
    print("--- 📝 Test Case 2: Requesting Document Data ---")
    query_2 = "Based on the uploaded project document, what database engine does Project Alpha use?"
    inputs_2 = {"messages": [HumanMessage(content=query_2)]}
    
    # Run the compiled LangGraph application again!
    output_2 = agent_app.invoke(inputs_2)
    
    print("\n🤖 Final Graph Response (Local RAG):")
    print(output_2["messages"][-1].content)
    print("\n📄 Retrieved Context Chunks saved in State:")
    for i, chunk in enumerate(output_2.get("context", []), 1):
        print(f"[{i}]: {chunk}")
    print("\n=============================================")