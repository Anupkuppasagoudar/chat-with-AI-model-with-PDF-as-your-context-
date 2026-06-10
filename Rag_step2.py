import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()
secret_key = os.getenv("GROQ_API_KEY")

# Initialize models and vector DB
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = FAISS.load_local("My_local_vector_db", embeddings_model, allow_dangerous_deserialization=True)
llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=secret_key, temperature=0, max_tokens=2048)

retriever = vector_db.as_retriever(search_kwargs={"k": 5})

# 1. Prompt for contextualizing the query (Notice we changed "history" to "chat_history")
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),  # Key alignment
        ("human", "{input}")
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# 2. Prompt for generating the actual answer
system_prompt = (
    "Answer the question based on the following context. "
    "Do not answer if you do not have the context to answer the question.\n\n"
    "Context:\n{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# 3. Create the master retrieval chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# This list will persist the context for the entire runtime session
chat_history = []

print("RAG System with Memory Initialized. Type 'exit' to quit.")
print("--------------------------------------------------")

while True:
    query = input("Enter your question: ")
    if query.lower() == 'exit':
        break
    
    if not query.strip():
        continue
        
    # The chain automatically handles history contextualization & document retrieval
    response = rag_chain.invoke({
        "input": query, 
        "chat_history": chat_history
    })
    
    print("--------------------------------------------------")
    print("Response:")
    print(response["answer"])
    print("--------------------------------------------------")
    
    # Crucial step: Append the current turn to chat history for subsequent iterations
    chat_history.extend([
        ("human", query),
        ("ai", response["answer"])
    ])