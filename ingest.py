from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import os

DATA_PATH = 'data/'
DB_CHROMA_PATH = 'vectorstore/db_chroma'

def main():
    for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            if file.endswith(".pdf"):
                print(file)
                loader = PyPDFLoader(os.path.join(root, file))
                documents = loader.load()
                print("Splitting into chunks")
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
                texts = text_splitter.split_documents(documents)
                # Create embeddings here
                print("Loading Ollama embeddings")
                embeddings = OllamaEmbeddings(model="nomic-embed-text")
                # Create vector store here
                print("Creating embeddings. This may take some time...")
                db = Chroma.from_documents(texts, embeddings, persist_directory=DB_CHROMA_PATH)
                db.persist()
                print("Ingestion complete! You can now query your documents.")

if __name__ == "__main__":
    main()
