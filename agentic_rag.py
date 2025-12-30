"""
Main agentic RAG system using CrewAI.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file FIRST, before any imports that need API keys
load_dotenv()

from agents import create_research_crew
from vector_store import create_or_load_vector_store


def initialize_system():
    """Initialize the vector store and verify setup."""
    print("Initializing Agentic RAG System...")
    print("=" * 50)
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not found in environment variables.")
        print("Please set it using: export OPENAI_API_KEY='your-key-here'")
        return False
    
    # Initialize vector store
    try:
        create_or_load_vector_store()
        print("System initialized successfully!")
        print("=" * 50)
        return True
    except Exception as e:
        print(f"Error initializing system: {e}")
        return False


def query_research_paper(query: str) -> str:
    """
    Query the research paper using agentic RAG.
    
    Args:
        query: The question or query about the research paper
    
    Returns:
        The answer from the agentic system
    """
    print(f"\n🔍 Processing query: {query}")
    print("=" * 50)
    
    # Create crew and execute
    crew = create_research_crew(query)
    result = crew.kickoff()
    
    return result


def interactive_mode():
    """Run the system in interactive mode."""
    if not initialize_system():
        return
    
    print("\n🤖 Agentic RAG System Ready!")
    print("Ask questions about the research paper. Type 'exit' to quit.\n")
    
    while True:
        query = input("\nYour question: ").strip()
        
        if query.lower() in ['exit', 'quit', 'q']:
            print("\nGoodbye!")
            break
        
        if not query:
            continue
        
        try:
            answer = query_research_paper(query)
            print("\n" + "=" * 50)
            print("📝 Answer:")
            print("=" * 50)
            print(answer)
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    interactive_mode()

