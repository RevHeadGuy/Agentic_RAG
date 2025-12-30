# Agentic RAG with CrewAI

An intelligent RAG (Retrieval-Augmented Generation) system using CrewAI that allows you to query a research paper using multiple AI agents working together.

## Features

- 🤖 **Multi-Agent System**: Uses CrewAI to coordinate multiple specialized agents
- 📚 **Vector Store**: ChromaDB for efficient document retrieval
- 🔍 **Intelligent Search**: Semantic search across the research paper
- 📝 **Comprehensive Answers**: Agents collaborate to provide detailed, well-structured answers

## Agents

1. **Research Analyst**: Searches and retrieves relevant information from the paper
2. **Content Summarizer**: Creates clear summaries of research findings
3. **Q&A Specialist**: Synthesizes information to provide comprehensive answers

## Setup

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Set your OpenAI API key**:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

Or create a `.env` file:
```
OPENAI_API_KEY=your-api-key-here
```

3. **Place your PDF**:
   - Ensure your PDF is in `data/RESEARCH PAPER.pdf`

## Usage

### Interactive Mode
Run the main script for an interactive query session:
```bash
python agentic_rag.py
```

### Programmatic Usage
```python
from agentic_rag import query_research_paper, initialize_system

# Initialize the system
initialize_system()

# Query the research paper
answer = query_research_paper("What is the main research question?")
print(answer)
```

### Example Script
Run the example script to see the system in action:
```bash
python example_usage.py
```

## Project Structure

```
.
├── agentic_rag.py      # Main agentic RAG system
├── agents.py           # CrewAI agents configuration
├── vector_store.py     # Vector store utilities and tools
├── rag_store.py        # Original vector store creation (optional)
├── example_usage.py    # Example usage script
├── requirements.txt    # Python dependencies
├── data/
│   └── RESEARCH PAPER.pdf
└── db/                 # ChromaDB vector store (created automatically)
```

## How It Works

1. **Document Processing**: The PDF is loaded, split into chunks, and embedded into a vector store
2. **Query Processing**: When you ask a question:
   - The Research Analyst searches the vector store for relevant sections
   - The Content Summarizer creates a summary of findings
   - The Q&A Specialist synthesizes everything into a comprehensive answer
3. **Collaborative Intelligence**: Agents work together, with each agent building on the previous one's work

## Notes

- The vector store is automatically created on first run and persisted for future use
- The system uses GPT-4 by default (configurable in `agents.py`)
- All agents have access to the search tool to query the research paper

