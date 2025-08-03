import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import  load_qa_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import google.generativeai as genai
import os
import asyncio
import google.auth
from google.oauth2 import service_account
from google.cloud import storage
import time

# Load service account info from secrets
service_account_info = st.secrets["gcp_service_account"]

# Create credentials from dict
credentials = service_account.Credentials.from_service_account_info(service_account_info)

# Initialize client with explicit credentials
client = storage.Client(credentials=credentials, project=service_account_info["project_id"])

# Get GEMINI_API_KEY
gemini_api_key = service_account_info["gemini_api_key"]

# Example usage (list buckets)
# buckets = list(client.list_buckets())
# st.write("Loaded secret keys:", list(service_account_info.keys()))
# st.write("Buckets:", buckets)
# st.write(gemini_api_key)

# credentials, project_id = google.auth.default()

# Option 1: Using environment variable (recommended)
genai.configure(api_key=gemini_api_key)

# Option 2: Passing API key directly (less secure for production)
# genai.configure(api_key=GEMINI_API_KEY)

# Extract the text
def get_pdf_text(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Break it into chunks
def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=500,
        # chunk_overlap=1000,
        # length_function=len,
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    # Create embeddings using a Google Generative AI model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Create a vector store using FAISS from the provided text chunks and embeddings
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)

    # Save the vector store locally with the name "faiss_index"
    # vector_store.save_local("faiss_index")

    return vector_store

def get_conversational_chain():
    # Initialize a ChatGoogleGenerativeAI model for conversational AI
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        temperature=0,
        max_tokens=2000,
        timeout=120,
        max_retries=3,
    )

    # Load a question-answering chain with the specified model and prompt
    chain = load_qa_chain(llm=llm, chain_type="stuff")

    return chain

def user_input(user_question, vector_store):
    # Perform similarity search in the vector database based on the user question
    match = vector_store.similarity_search(user_question)

    # Obtain a conversational question-answering chain
    chain = get_conversational_chain()

    # Use the conversational chain to get a response based on the user question and retrieved documents
    response = chain.run(input_documents=match, question=user_question)

    # Display the response in a Streamlit app (assuming 'st' is a Streamlit module)
    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config("Chatterly")
    # Upload PDF files
    st.header("Chatterly")

    with st.sidebar:
        st.title("Document")
        file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if file is not None:
        with st.spinner("Processing..."):
            raw_text = get_pdf_text(file)
            text_chunks = get_chunks(raw_text)
            vector_store = get_vector_store(text_chunks)

            user_question = st.text_input("Type your question here")

            if user_question:
                user_input(user_question, vector_store)

if __name__ == "__main__":
    main()