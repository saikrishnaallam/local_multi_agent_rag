from typing import List, TypedDict, Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph, START
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import Chroma
import os

# --- 1. Define the Graph State ---
class GraphState(TypedDict):
    """Represents the state of our graph."""
    question: str
    documents: List[str]
    generation: str
    route: str  # Added to track step route

# --- 2. Define the Guardrail (Pydantic) ---
class RouteDecision(BaseModel):
    """Route the process based on document relevance."""
    step: Literal["generate", "web_search"] = Field(
        description="Output 'generate' if documents are relevant to the question. Output 'web_search' if they are irrelevant."
    )

# Initialize local LLM, Embeddings, Search Tool
llm = ChatOllama(model="llama3", temperature=0)
web_search_tool = DuckDuckGoSearchRun()
local_embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_db = None

def get_vector_db():
    global vector_db
    if vector_db is None:
        persist_dir = "./chroma_db"
        if os.path.exists(persist_dir):
            print(f"📦 Connecting to persistent Chroma DB at '{persist_dir}'...")
            vector_db = Chroma(persist_directory=persist_dir, embedding_function=local_embeddings)
        else:
            print(f"⚠️ Warning: Directory '{persist_dir}' not found. Please run ingest.py first.")
    return vector_db

# --- 3. Define the Nodes ---

def retrieve_node(state: GraphState):
    """Retrieves documents from your vector store."""
    question = state["question"]
    print(f"\n🔎 Retrieving documents from vector store for query: '{question}'...")
    
    import streamlit as st
    db = None
    if "vector_db" in st.session_state and st.session_state.vector_db is not None:
        db = st.session_state.vector_db
    else:
        db = get_vector_db()
        
    if db is not None:
        retriever = db.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(question)
        doc_texts = [d.page_content for d in docs]
    else:
        print("⚠️ Vector store not initialized. Returning empty list.")
        doc_texts = []
        
    return {"documents": doc_texts, "question": question}

def grade_documents_node(state: GraphState):
    """Evaluates if the retrieved documents are relevant to the question."""
    question = state["question"]
    documents = state["documents"]
    
    print("⚖️ Grading retrieved documents for relevance...")
    if not documents:
        print("⚠️ No documents retrieved. Routing directly to web search.")
        return {"route": "web_search"}
        
    # Set up the LLM with the strict output guardrail
    structured_llm = llm.with_structured_output(RouteDecision)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert grader assessing the relevance of retrieved document chunks to a user question.\n"
            "Your task is to determine whether the provided document context contains information directly relevant to answering the user question.\n"
            "If the documents contain useful facts to answer the question, step must be 'generate'.\n"
            "If the documents do not contain relevant information, step must be 'web_search'.\n"
            "Be strict. If the document is about a different topic, output 'web_search'."
        )),
        ("human", "Retrieved document context: \n\n {document} \n\n User question: {question}")
    ])
    
    grader_chain = prompt | structured_llm
    
    # Combine docs for grading
    doc_text = "\n\n".join(documents)
    
    try:
        # The LLM is forced to return an object matching RouteDecision
        decision = grader_chain.invoke({"document": doc_text, "question": question})
        route_step = decision.step
    except Exception as e:
        print(f"❌ Structured grading failed: {e}. Defaulting to 'web_search'.")
        route_step = "web_search"
        
    print(f"🎯 Grader Decision: Directed to -> {route_step}")
    # We store the decision in the state (or pass it directly to the router)
    return {"route": route_step}

def generate_node(state: GraphState):
    """Generates the final answer using the relevant documents."""
    question = state["question"]
    documents = state["documents"]
    
    print("🤖 Generating the final answer using local documents...")
    context = "\n\n".join(documents)
    
    prompt = f"""You are a helpful assistant. Use the following context to answer the user's query.

Context:
{context}

Question:
{question}"""
    
    response = llm.invoke(prompt)
    return {"generation": response.content}

def web_search_node(state: GraphState):
    """Falls back to web search if local docs fail."""
    question = state["question"]
    documents = state["documents"]
    
    print(f"🌐 Falling back to live Web Search for: '{question}'...")
    try:
        search_results = web_search_tool.run(question)
    except Exception as e:
        search_results = f"Search failed or timed out. Error: {e}"
        
    return {"documents": documents + [search_results]}

# --- 4. Define the Router Function ---

def route_question(state: GraphState) -> Literal["generate", "web_search"]:
    """Reads the LLM's decision and routes the graph."""
    # We call the grader node's logic here to get the route
    decision = grade_documents_node(state)
    return decision["route"]

# --- 5. Build and Compile the Graph ---

workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_node("web_search", web_search_node)

# Set the entry point
workflow.set_entry_point("retrieve")

# Add a conditional edge right after retrieval
workflow.add_conditional_edges(
    "retrieve",
    route_question,
    {
        "generate": "generate",     # If router returns "generate", go to generate node
        "web_search": "web_search"  # If router returns "web_search", go to web search node
    }
)

# Connect the rest of the flow
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

# Compile!
app = workflow.compile()

if __name__ == "__main__":
    print("\n=============================================")
    print("🚀 RUNNING CORRECTIVE RAG (CRAG) MULTI-AGENT SYSTEM")
    print("=============================================\n")
    
    # Test Case 1: Ingested details (Expected: Route to 'generate')
    print("--- 📝 Test Case 1: Ask about Project Lead (In-Context) ---")
    inputs_1 = {"question": "Who is the project lead for Project Alpha?"}
    output_1 = app.invoke(inputs_1)
    print("\n🤖 Final Output (Test Case 1):")
    print(output_1["generation"])
    print("\n---------------------------------------------\n")
    
    # Test Case 2: Out of context details (Expected: Route to 'web_search')
    print("--- 📝 Test Case 2: Ask about Super Bowl Winner (Out-of-Context) ---")
    inputs_2 = {"question": "Who won the latest Super Bowl and what was the score?"}
    output_2 = app.invoke(inputs_2)
    print("\n🤖 Final Output (Test Case 2):")
    print(output_2["generation"])
    print("\n=============================================")
