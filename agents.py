"""
CrewAI agents configuration for agentic RAG system.
"""
from dotenv import load_dotenv
import os

# Load environment variables (in case this module is imported directly)
load_dotenv()

from crewai import Agent, Task, Crew
from crewai.tools import BaseTool
from langchain_openai import ChatOpenAI
from vector_store import search_research_paper


# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4o",  # Using gpt-4o (more accessible) - can also use "gpt-3.5-turbo" or "gpt-4-turbo"
    temperature=0.7
)


# Create a custom tool class for CrewAI
class ResearchPaperSearchTool(BaseTool):
    name: str = "search_research_paper"
    description: str = "Search the research paper PDF for relevant information based on a query. Use this tool to find specific information, facts, or sections from the research paper."
    
    def _run(self, query: str) -> str:
        """Execute the search."""
        return search_research_paper(query)


# Create tool instance
search_tool = ResearchPaperSearchTool()

# Define agents
researcher_agent = Agent(
    role='Research Analyst',
    goal='Search and retrieve relevant information from the research paper based on user queries',
    backstory="""You are an expert research analyst with deep knowledge in academic papers.
    Your specialty is finding and extracting precise information from research documents.
    You excel at understanding context and retrieving the most relevant sections.""",
    tools=[search_tool],
    verbose=True,
    allow_delegation=False,
    llm=llm
)


summarizer_agent = Agent(
    role='Content Summarizer',
    goal='Create clear, concise summaries of research paper content',
    backstory="""You are a skilled technical writer who specializes in summarizing
    complex research content into clear, understandable summaries. You maintain
    accuracy while making information accessible.""",
    verbose=True,
    allow_delegation=False,
    llm=llm
)


qa_agent = Agent(
    role='Q&A Specialist',
    goal='Answer questions comprehensively based on research paper content',
    backstory="""You are an expert at answering questions based on research documents.
    You provide accurate, well-structured answers by synthesizing information from
    multiple sources within the document. You always cite your sources and provide
    context for your answers.""",
    verbose=True,
    allow_delegation=False,
    llm=llm
)


def create_research_crew(query: str) -> Crew:
    """
    Create a crew for research tasks.
    
    Args:
        query: The user's query about the research paper
    
    Returns:
        A configured Crew instance
    """
    research_task = Task(
        description=f"""
        Search the research paper for information related to: {query}
        
        Find all relevant sections, facts, and details that answer or relate to this query.
        Be thorough and comprehensive in your search.
        """,
        agent=researcher_agent,
        expected_output="A detailed list of relevant information, facts, and sections from the research paper"
    )
    
    summarize_task = Task(
        description="""
        Summarize the research findings provided by the Research Analyst.
        Create a clear, well-organized summary that highlights the key points.
        """,
        agent=summarizer_agent,
        expected_output="A clear and concise summary of the research findings"
    )
    
    answer_task = Task(
        description=f"""
        Based on the research findings and summary, provide a comprehensive answer to: {query}
        
        Your answer should:
        1. Directly address the query
        2. Include relevant details and context
        3. Cite specific information from the research paper
        4. Be well-structured and easy to understand
        """,
        agent=qa_agent,
        expected_output="A comprehensive, well-structured answer to the user's query with citations"
    )
    
    crew = Crew(
        agents=[researcher_agent, summarizer_agent, qa_agent],
        tasks=[research_task, summarize_task, answer_task],
        verbose=True
    )
    
    return crew

