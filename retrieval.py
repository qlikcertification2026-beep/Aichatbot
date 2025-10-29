# import basics
import os
from dotenv import load_dotenv

# vector store
from langchain_chroma import Chroma

# import langchain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

load_dotenv()

# initialize embeddings and local Chroma vector store (Gemini)
embeddings = GoogleGenerativeAIEmbeddings(
    model=os.environ.get("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004"),
    google_api_key=os.environ.get("GOOGLE_API_KEY"),
    transport="rest",
)
vector_store = Chroma(persist_directory="chroma_db_gemini", embedding_function=embeddings)

# retrieval
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2},
)
results = retriever.invoke("what is retrieval augmented generation?")

# show results
print("RESULTS:")

for res in results:
    print(f"* {res.page_content} [{res.metadata}]")