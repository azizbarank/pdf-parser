import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import os

from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

# Streamlit App Configuration
st.set_page_config(page_title="PDF Query Assistant", layout="centered")
st.title("Langchain PDF Query Assistant")

# User Input: API Key
openai_api_key = st.text_input("Enter your OpenAI API key:", type="password")
if openai_api_key:
    # Set the OpenAI API key for the session
    os.environ["OPENAI_API_KEY"] = openai_api_key

# Initialize embeddings (based on user input API key)
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)


def create_db_from_pdf(pdf_path):
    """Create a vectorstore database from a PDF file using FAISS."""
    loader = PyMuPDFLoader(pdf_path)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    # Initialize FAISS vector store
    db = FAISS.from_documents(docs, embeddings)
    return db


def get_response_from_query(db, query, k=4):
    """Get a response from the vectorstore based on a query."""
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    chat = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0.2)

    template = """
        You are a helpful assistant that can answer questions about PDF files 
        based on the transcript of these files: {docs}
        Only use factual information from the transcript to answer the question.
        If you feel like you don't have enough information to answer the question, say "I don't know".
    """
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    human_template = "Answer the following question: {question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )
    chain = LLMChain(llm=chat, prompt=chat_prompt)

    response = chain.run(question=query, docs=docs_page_content)
    return response.strip()


# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    pdf_path = f"file-path.pdf"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Processing the PDF and creating a database..."):
        try:
            db = create_db_from_pdf(pdf_path)
            st.success("PDF successfully processed!")

            query = st.text_input("Ask a question about the PDF:")
            if query:
                with st.spinner("Fetching the answer..."):
                    response = get_response_from_query(db, query)
                    st.subheader("Response:")
                    st.write(response)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
else:
    st.info("Please upload a PDF to get started.")
