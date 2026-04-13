import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

def build_vector_store(docs_dir, store_path):
    """
    Loads all PDFs from a directory, chunks them, and builds a FAISS index.
    """
    documents = []
    for filename in os.listdir(docs_dir):
        if filename.endswith(".pdf"):
            file_path = os.path.join(docs_dir, filename)
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
    
    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)
    
    # Embeddings and Vector Store
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # Save the index
    vector_store.save_local(store_path)
    print(f"Vector store saved to {store_path}")
    return vector_store

def load_vector_store(store_path):
    """
    Loads an existing FAISS index.
    """
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)

if __name__ == "__main__":
    docs_dir = "data/raw_docs"
    store_path = "data/vector_store/faiss_index"
    build_vector_store(docs_dir, store_path)
