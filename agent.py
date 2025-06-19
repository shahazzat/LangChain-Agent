from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Initialize LLM (using Mistral locally via Ollama)
llm = Ollama(model="mistral")

# Add conversation memory
memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory)

# Chat loop
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    response = conversation.invoke({"input": user_input})
    print(f"Agent: {response['response']}")