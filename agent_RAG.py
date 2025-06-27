from langchain.agents import AgentExecutor, Tool, create_react_agent
from langchain_community.llms import Ollama
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from bs4 import BeautifulSoup

# Add caching for embeddings
from langchain.cache import SQLiteCache
from langchain.globals import set_llm_cache
set_llm_cache(SQLiteCache(database_path=".langchain.db"))


llm = Ollama(model="mistral")

# Load a webpage
loader = WebBaseLoader("https://en.wikipedia.org/wiki/Large_language_model")
docs = loader.load()

# Split into chunks
# Update the text splitter with better parameters
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,  # Better for most LLM contexts
    chunk_overlap=128,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]  # More natural splitting
)

# Add document cleaning
def clean_html(docs):
    for doc in docs:
        soup = BeautifulSoup(doc.page_content, 'html.parser')
        doc.page_content = soup.get_text(separator=' ', strip=True)
    return docs

cleaned_docs = clean_html(docs)
splits = text_splitter.split_documents(cleaned_docs)

# Create vector store
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = FAISS.from_documents(splits, embeddings)

## Hybrid Document Search
# Add keyword-based retriever
bm25_retriever = BM25Retriever.from_documents(splits)
bm25_retriever.k = 2  # Number of keyword results

# Configure vector retriever
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Combine both methods
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6]  # Tune based on your needs
)

# Update tool
tools = [
    Tool(
        name="Hybrid Document Search",
        func=ensemble_retriever.get_relevant_documents,
        description="Combines semantic and keyword search for better results"
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
question = "What are the risks of LLMs?"
response = agent_executor.invoke({"input": question})
print(response["output"])

