# LangChain-Agent (Local LLM Agent with Document Retrieval)

A LangChain-based AI agent that answers questions using a local Mistral LLM (via Ollama) and retrieves information from documents.

## Features

- Local LLM execution using Ollama with Mistral model
- Web document loading and processing
- Vector embeddings with FAISS
- Custom ReAct-style prompt engineering
- Document-based question answering

## Prerequisites

- Python 3.8+
- Ollama installed locally
- At least 8GB RAM (16GB recommended)

## Installation

1. **Install Ollama**:
   ```bash
   curl -fsSL https://ollama.ai/install.sh | sh
   ```

2. **Download the required LLM models**:
   ```bash
   ollama pull mistral
   ollama pull nomic-embed-text
   ```

3. **Set up Python environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or 
   venv\Scripts\activate    # Windows
   ```

4. **Install Python dependencies**:
   ```bash
   pip install langchain langchain-community faiss-cpu beautifulsoup4
   ```

## Configuration

1. Ensure Ollama service is running:
   ```bash
   ollama serve
   ```

2. Modify the following variables in the code as needed:
   - `model="mistral"` - Change to use different Ollama models
   - `WebBaseLoader` URL - Change to load different documents
   - `chunk_size` and `chunk_overlap` - Adjust for document processing

## Usage

1. Run the agent script:
   ```bash
   python agent_RAG.py
   ```

2. The agent will:
   - Load and process the Wikipedia page about LLMs
   - Create a vector store for document retrieval
   - Answer questions using both the LLM and document knowledge

Example question handled by default:
```python
response = agent_executor.invoke({"input": "What are the risks of LLMs?"})
```

## Customization

### Using Different Models
Replace in code:
```python
llm = Ollama(model="mistral")  # Try "llama2", "neural-chat", etc.
embeddings = OllamaEmbeddings(model="nomic-embed-text")  # Try "llama2", "all-minilm"
```

### Adding More Tools
Extend the tools list:
```python
tools.append(
    Tool(
        name="Web Search",
        func=search_tool,
        description="Useful for current events"
    )
)
```

### Modifying the Prompt
Edit the `custom_react_prompt` string to change agent behavior.

## Troubleshooting

### "Model not found" errors
Ensure you've pulled the models:
```bash
ollama pull mistral
ollama pull nomic-embed-text
```

### Performance issues
- Use quantized models (e.g., `mistral:7b-instruct-q4`)
- Reduce chunk sizes in `RecursiveCharacterTextSplitter`
- Use CPU-optimized embeddings like `all-MiniLM-L6-v2`

## License

MIT License

## Acknowledgments

- LangChain team
- Ollama developers
- Mistral AI
```


### Recommended Repository Structure:
```
/llm-agent
│
├── agent_RAG.py          # Main agent code
├── README.md         # This documentation
├── requirements.txt  # Python dependencies
└── /data             # (Optional) For local documents
```

To create the `requirements.txt` file:
```bash
pip freeze > requirements.txt
```
