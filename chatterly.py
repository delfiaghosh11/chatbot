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

# Upload PDF files
st.header("Chatterly")

with st.sidebar:
    st.title("Documents")
    file = st.file_uploader("Choose a PDF file", type=["pdf"])

# Extract the text
if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
        # st.write(text)

    # Break it into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=500,
        # chunk_overlap=1000,
        length_function=len,
    )
    with st.spinner("Chunking...", show_time=False):
        time.sleep(5)
    chunks = text_splitter.split_text(text)
    # st.write(chunks)

    # Generate Embeddings
    # If no event loop is present, set one
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        with st.spinner("Embedding...", show_time=False):
            time.sleep(5)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", task_type="retrieval_document")

        # Create Vector Store
        # 1. Generating Embeddings
        # 2. Initializing the FAISS dB
        # 3. Storing the chunks & embeddings in the dB

        # Create a vector store using FAISS from the provided text chunks and embeddings
        with st.spinner("Storing...", show_time=False):
            time.sleep(5)
        vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)

        # Save the vector store locally with the name "faiss_index"
        # vector_store.save_local("faiss_index")

        # Get user's questions
        user_question = st.text_input("Type your question here")

        # Do similarity search on it
        if user_question:
            with st.spinner("Searching...", show_time=False):
                time.sleep(5)
            match = vector_store.similarity_search(user_question)
            # st.write(match)

            # Output the results
            # 1. Chain
            # 2. Take question
            # 3. Get relevant document
            # 4. Pass it to the LLM
            # 5. Generate the output

            # Define LLM
            # temperature value is used to define if we want the llm to generate random answers or, be specific
            with st.spinner("Almost there!", show_time=False):
                time.sleep(5)
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-pro",
                temperature=0,
                max_tokens=2000,
                timeout=120,
                max_retries=3,
            )

            # Output the result
            with st.spinner("Almost there!", show_time=False):
                time.sleep(5)
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=match, question=user_question)
            st.write(response)