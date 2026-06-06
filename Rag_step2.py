import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

load_dotenv()
#secret_key = os.getenv("Gemini_api_key")
secret_key = os.getenv("GROQ_API_KEY")
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=secret_key) 
vector_db = FAISS.load_local("My_local_vector_db", embeddings_model, allow_dangerous_deserialization=True)
#llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=secret_key)
llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=secret_key, temperature=0, max_tokens=2048)



def GetResponse(query):
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in docs])
    prompt = f"Answer the question based on the following context: {context}\nQuestion: {query}, Do not answer if you do not have the context to answer the question."
    response = llm.invoke(prompt)
    return response.content

while True:
    query = input("Enter your question (or 'exit' to quit): ")
    if query.lower() == 'exit':
        break
    response = GetResponse(query)
    print("--------------------------------------------------")
    print("Response:")
    print(response)





