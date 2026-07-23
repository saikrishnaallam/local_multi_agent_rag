import streamlit as st
import tempfile
import os
from main import app as compiled_graph, local_embeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

st.set_page_config(page_title="Multi-Agent RAG", page_icon="🤖")
st.title("Interactive RAG Assistant 🤖")

# --- 1. Memory & Vector DB Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_db" not in st.session_state:
    if os.path.exists("./chroma_db"):
        st.session_state.vector_db = Chroma(persist_directory="./chroma_db", embedding_function=local_embeddings)
    else:
        st.session_state.vector_db = None

# --- 2. Sidebar for File Uploads ---
with st.sidebar:
    st.header("Document Upload 📄")
    uploaded_files = st.file_uploader("Drop your PDFs here", type=["pdf"], accept_multiple_files=True)
    
    if st.button("Process Documents") and uploaded_files:
        with st.spinner("Processing documents into Vector Store..."):
            # Ensure the vector database is initialized in session state
            if st.session_state.vector_db is None:
                st.session_state.vector_db = Chroma(persist_directory="./chroma_db", embedding_function=local_embeddings)
            
            try:
                # Clear all old document IDs to prevent mixing old context with new uploads
                all_ids = st.session_state.vector_db.get()["ids"]
                if all_ids:
                    st.session_state.vector_db.delete(all_ids)
            except Exception as e:
                st.warning(f"Could not clear old vector database: {e}")
            
            total_chunks = 0
            for uploaded_file in uploaded_files:
                # Save uploaded file to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                
                try:
                    # Load and split document
                    loader = PyPDFLoader(tmp_path)
                    pages = loader.load()
                    if not pages:
                        st.warning(f"Could not extract any pages from '{uploaded_file.name}'. Is it empty or scanned?")
                        continue
                        
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                    chunks = text_splitter.split_documents(pages)
                    
                    if chunks:
                        # Add to vector database
                        st.session_state.vector_db.add_documents(chunks)
                        total_chunks += len(chunks)
                    else:
                        st.warning(f"No text chunks could be generated from '{uploaded_file.name}'.")
                except Exception as e:
                    st.error(f"Error processing '{uploaded_file.name}': {e}")
                finally:
                    # Clean up temp file
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                        
            st.success(f"Successfully processed {len(uploaded_files)} document(s) ({total_chunks} chunks indexed)!")

# --- 3. Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 4. Chat Input & Visual Tracing ---
if prompt := st.chat_input("Ask a question about your documents..."):
    
    # Save and display the user's prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display the assistant's response
    with st.chat_message("assistant"):
        
        # st.status creates an expandable box to show our agent's steps
        with st.status("Agent Workflow 🕵️‍♂️", expanded=True) as status:
            st.write("Starting...")
            
            initial_state = {"question": prompt}
            final_generation = ""
            
            # Run the compiled graph streaming output
            for output in compiled_graph.stream(initial_state):
                for node_name, node_state in output.items():
                    # Visual trace: Tell the user which node just ran!
                    if node_name == "retrieve":
                        st.write("🔍 Retrieving context chunks from vector store...")
                    elif node_name == "web_search":
                        st.write("🌐 Fallback: Searching the web...")
                    elif node_name == "generate":
                        st.write("🤖 Generating answer...")
                    else:
                        st.write(f"✅ Node completed: `{node_name}`")
                    
                    # Capture the final generation if it exists
                    if "generation" in node_state and node_state["generation"]:
                        final_generation = node_state["generation"]
            
            if not final_generation:
                final_generation = "Sorry, no response could be generated."
                
            status.update(label="Response Ready!", state="complete", expanded=False)
            
        # Display the final answer outside the status box
        st.markdown(final_generation)
        st.session_state.messages.append({"role": "assistant", "content": final_generation})