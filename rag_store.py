from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

def create_vector_store():
    loader = PyPDFLoader("data/RESEARCH PAPER.pdf")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()

    vectordb = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory="db"
    )
    vectordb.persist()
    return vectordb 