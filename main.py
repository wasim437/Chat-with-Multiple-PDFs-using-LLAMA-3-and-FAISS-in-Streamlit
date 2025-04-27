# Import Libraries
import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


# Load environment variables
load_dotenv()

# Define working directory
working_dir = os.getcwd()

# Helper Functions
def load_documents(file_paths):
    all_documents = []
    for path in file_paths:
        loader = PyMuPDFLoader(path)
        documents = loader.load()
        all_documents.extend(documents)
    return all_documents

def setup_vectorstore(documents):
    embeddings = HuggingFaceEmbeddings()
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200
    )
    doc_chunks = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(doc_chunks, embeddings)
    return vectorstore

def create_chain(vectorstore):
    llm = ChatGroq(
        model="llama3-70b-8192",
        temperature=0,
        groq_api_key="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"
    )
    retriever = vectorstore.as_retriever()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=True
    )
    return chain

# Streamlit App UI
st.set_page_config(page_title="Multi-PDF Chat", page_icon="📄", layout="centered")
st.title("📄 Chat with Multiple PDFs using LLAMA 3")

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = None

# Upload and Process Multiple PDFs
uploaded_files = st.file_uploader("Upload one or more PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    file_paths = []

    # Save files temporarily
    for uploaded_file in uploaded_files:
        file_path = os.path.join(working_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        file_paths.append(file_path)

    if st.session_state.vectorstore is None:
        documents = load_documents(file_paths)
        st.session_state.vectorstore = setup_vectorstore(documents)

    if st.session_state.conversation_chain is None:
        st.session_state.conversation_chain = create_chain(st.session_state.vectorstore)

# Display Chat History
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
user_input = st.chat_input("Ask something about the documents...")

if user_input and st.session_state.conversation_chain:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        try:
            response = st.session_state.conversation_chain.invoke({"question": user_input})
            assistant_response = response["answer"]
        except Exception as e:
            assistant_response = f"❌ Error: {e}"

        st.markdown(assistant_response)
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
