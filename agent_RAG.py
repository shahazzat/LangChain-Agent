from langchain.agents import AgentExecutor, Tool, create_react_agent
from langchain_community.llms import Ollama
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate

llm = Ollama(model="mistral")

# Load a webpage
loader = WebBaseLoader("https://en.wikipedia.org/wiki/Large_language_model")
docs = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Create vector store
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = FAISS.from_documents(splits, embeddings)

# Add as a tool
retriever = vectorstore.as_retriever()
tools = [
    Tool(
        name="Document Search",
        func=retriever.get_relevant_documents,
        description="Useful for answering questions about LLMs"
    )
]

# Create agent
custom_react_prompt = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

prompt_template = PromptTemplate.from_template(custom_react_prompt)
agent = create_react_agent(llm, tools, prompt=prompt_template)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Now the agent can use docs for answers!
response = agent_executor.invoke({"input": "What are the risks of LLMs?"})
print(response["output"])