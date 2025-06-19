from langchain.agents import AgentExecutor, Tool, create_react_agent
from langchain_community.llms import Ollama
from langchain import hub
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import WolframAlphaAPIWrapper

# Initialize LLM
# llm = Ollama(model="llama3")
llm = Ollama(model="mistral")

# Define tools
search = DuckDuckGoSearchRun()
wolfram = WolframAlphaAPIWrapper()  # Requires API key [export WOLFRAM_ALPHA_APPID=ACXR-JYHG] (https://developer.wolframalpha.com/) 

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="Useful for finding current information"
    ),
    Tool(
        name="Calculator",
        func=wolfram.run,
        description="Useful for math/science questions"
    )
]

# Create agent
prompt = hub.pull("hwchase17/react")  # Pre-built prompt template https://smith.langchain.com/hub/hwchase17/react?tab=0
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Run the agent
response = agent_executor.invoke({"input": "What's the population of Japan divided by 2?"})
print(response["output"])