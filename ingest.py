from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

def load_and_index_pdf(file_path: str):
    # 1. Load the PDF
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    # 2. Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(pages)

    # 3. Embed and store in ChromaDB
    # Note: Ensure you have pulled the embedding model in Ollama (e.g., 'nomic-embed-text' or 'llama3')
    embeddings = OllamaEmbeddings(model="nomic-embed-text") 
    
    # Create and persist the vector database locally
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    return vector_db

if __name__ == "__main__":
    import sys
    import os
    
    file_path = "project_alpha_overview.pdf"
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        
    if not os.path.exists(file_path):
        print(f"❌ Error: File '{file_path}' does not exist.")
        sys.exit(1)
        
    print(f"📦 Starting indexing of '{file_path}'...")
    db = load_and_index_pdf(file_path)
    print("✅ Ingestion successfully completed! Vector database stored in './chroma_db'.")