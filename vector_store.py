"""
Vector store utility for CrewAI agents to access the PDF document.
"""
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from typing import List
import os


def create_or_load_vector_store():
    """Create or load the vector store from the PDF."""
    persist_directory = "db"
    
    # Check if vector store already exists
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        print("Loading existing vector store...")
        embeddings = OpenAIEmbeddings()
        vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
    else:
        print("Creating new vector store...")
        loader = PyPDFLoader("data/RESEARCH PAPER.pdf")
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        chunks = splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings()

        vectordb = Chroma.from_documents(
            chunks,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        vectordb.persist()
        print(f"Vector store created with {len(chunks)} chunks")
    
    return vectordb


# Global vector store instance
_vector_store = None


def get_vector_store():
    """Get or create the vector store instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = create_or_load_vector_store()
    return _vector_store


def search_research_paper(query: str) -> str:
    """
    Search the research paper PDF for relevant information.
    
    Args:
        query: The search query to find relevant information in the paper.
    
    Returns:
        A string containing the most relevant chunks from the research paper.
    """
    vectordb = get_vector_store()
    
    # Perform similarity search
    results = vectordb.similarity_search(query, k=5)
    
    # Combine results into a readable format
    context = "\n\n---\n\n".join([
        f"Chunk {i+1}:\n{doc.page_content}\n[Source: Page {doc.metadata.get('page', 'unknown')}]"
        for i, doc in enumerate(results)
    ])
    
    return f"Found {len(results)} relevant sections:\n\n{context}"

