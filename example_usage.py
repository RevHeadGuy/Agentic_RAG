from agentic_rag import query_research_paper, initialize_system


def main():
    # Initialize the system
    if not initialize_system():
        print("Failed to initialize system. Please check your setup.")
        return
    
    # Example queries
    example_queries = [
        "What is the main research question or hypothesis of this paper?",
        "What methodology was used in this research?",
        "What are the key findings or results?",
        "What are the main conclusions of this paper?",
    ]
    
    print("\n" + "=" * 70)
    print("Example Queries - Agentic RAG System")
    print("=" * 70)
    
    for i, query in enumerate(example_queries, 1):
        print(f"\n\nExample {i}:")
        print(f"Query: {query}")
        print("-" * 70)
        
        try:
            answer = query_research_paper(query)
            print(f"\nAnswer:\n{answer}")
        except Exception as e:
            print(f"Error: {e}")
        
        print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

