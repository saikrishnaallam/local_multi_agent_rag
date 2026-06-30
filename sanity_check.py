from langchain_ollama import ChatOllama

print("🔄 Connecting to local Llama3 model via Ollama...")

try:
    # Initialize the local model connection
    # We set temperature=0 for precise, deterministic agent behaviors later
    llm = ChatOllama(model="llama3", temperature=0)
    
    # Test message
    response = llm.invoke("Confirm you are running locally on a Mac M2.")
    
    print("\n✅ Success! Your local AI brain is online.")
    print(f"🤖 AI Response: {response.content}")

except Exception as e:
    print("\n❌ Connection Failed.")
    print(f"Error Details: {e}")
    print("Tip: Ensure the Ollama app is open and running in your Mac's menu bar.")