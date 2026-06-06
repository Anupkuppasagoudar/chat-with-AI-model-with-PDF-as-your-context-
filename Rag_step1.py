from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv


load_dotenv()
secret_key = os.getenv("Gemini_api_key")
embeddings_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=secret_key)

print("Loading PDF document...")
loader = PyPDFLoader("./angular.pdf")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50, separators=["\n\n", "\n", " ", ""])
chunks = splitter.split_documents(documents)
vector_db = FAISS.from_documents(chunks, embeddings_model)
vector_db.save_local("My_local_vector_db")





    

    
