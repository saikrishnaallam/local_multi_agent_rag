from langchain_ollama import ChatOllama, OllamaEmbeddings

print("🔄 Connecting to local Llama3 model via Ollama...")

try:
    # Initialize the local model connection
    # We set temperature=0 for precise, deterministic agent behaviors later
    llm = ChatOllama(model="llama3", temperature=0)
    
    # Test message
    response = llm.invoke("Confirm you are running locally on a Mac M2.")
    
    print("\n✅ Success! Your local LLM is online.")
    print(f"🤖 AI Response: {response.content}")

    print("\n🔄 Connecting to local nomic-embed-text model via Ollama...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    test_vector = embeddings.embed_query("Sanity check embedding test query.")
    print(f"✅ Success! Your local Embedding model is online (Dimension: {len(test_vector)}).")
    print("\n🎉 All local AI brains are online and ready!")

except Exception as e:
    print("\n❌ Connection Failed.")
    print(f"Error Details: {e}")
    print("Tips: \n1. Ensure the Ollama app is open and running in your Mac's menu bar.\n2. Verify you have pulled both models: 'ollama pull llama3' and 'ollama pull nomic-embed-text'.")