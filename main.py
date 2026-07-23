from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Literal
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import Chroma
import os

# 1. Define the Shared Memory (State)
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    next_step: str

# 2. Initialize the Local Brain, Search Tool, and Local Embeddings
local_llm = ChatOllama(model="llama3", temperature=0)
web_search_tool = DuckDuckGoSearchRun()

# Use nomic-embed-text to match the embeddings of ingest.py
local_embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Connect to the local, persistent Chroma vector database
persist_dir = "./chroma_db"
if os.path.exists(persist_dir):
    print(f"📦 Connecting to persistent Chroma DB at '{persist_dir}'...")
    vector_db = Chroma(persist_directory=persist_dir, embedding_function=local_embeddings)
else:
    print(f"⚠️ Warning: Directory '{persist_dir}' not found. Please run ingest.py first.")
    vector_db = None

# --- Supervisor Router Node ---
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

# --- Web Search Agent Node ---
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

# --- Local RAG Agent Node ---
def rag_agent(state: AgentState):
    print("\n📄 Local RAG Agent is activating...")
    global vector_db
    
    user_query = state["messages"][-1].content
    print(f"🔎 Querying local vector database for: '{user_query}'...")
    
    if vector_db is None:
        # Try to reload on the fly in case it was created after startup
        persist_dir = "./chroma_db"
        if os.path.exists(persist_dir):
            print(f"📦 Connecting to persistent Chroma DB at '{persist_dir}'...")
            vector_db = Chroma(persist_directory=persist_dir, embedding_function=local_embeddings)
        else:
            return {"messages": [AIMessage(content="Error: Persistent vector database not initialized. Run ingest.py first.")]}
    
    # Retrieve top 3 most relevant document chunks
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(user_query)
    
    # Combine the text chunks into a single context string
    retrieved_context = "\n\n".join([d.page_content for d in docs])
    
    # Formulate system prompt
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
    
    print("🤖 Local model is generating a response based on document data...")
    ai_response = local_llm.invoke([system_message, combined_input])
    
    return {"messages": [AIMessage(content=ai_response.content)]}

# 1. Initialize the StateGraph
workflow = StateGraph(AgentState)

# 2. Add the execution nodes to the graph
workflow.add_node("web_search", web_search_agent)
workflow.add_node("local_rag", rag_agent)

# 3. Set up conditional routing logic
workflow.add_conditional_edges(
    START,
    supervisor_router,
    {
        "search": "web_search",
        "rag": "local_rag"
    }
)

# 4. Connect the output of both agents to the END node
workflow.add_edge("web_search", END)
workflow.add_edge("local_rag", END)

# 5. Compile the graph
local_agent_app = workflow.compile()

if __name__ == "__main__":
    print("\n=============================================")
    print("🚀 RUNNING PERSISTENT MULTI-AGENT SYSTEM")
    print("=============================================\n")
    
    # Test Case: Querying the document details we just ingested
    print("--- 📝 Test Case: Querying Document Data (RAG) ---")
    query = "Based on the uploaded project document, who is the project lead for Project Alpha?"
    inputs = {"messages": [HumanMessage(content=query)]}
    
    output = local_agent_app.invoke(inputs)
    
    print("\n🤖 Final Graph Response (Local RAG):")
    print(output["messages"][-1].content)
    print("\n=============================================")
