# Chat-with-Multiple-PDFs-using-LLAMA-3-and-FAISS-in-Streamlit

Project Title
Chat with Multiple PDFs using LLAMA 3 and FAISS in Streamlit

Project Summary
This project allows users to upload multiple PDF documents and interact with them through a chat interface. It uses a powerful AI model (LLAMA 3) to understand and answer questions based on the uploaded PDFs. The project is built using Streamlit for the user interface, FAISS for efficient document search, and Hugging Face Embeddings to process text into meaningful vectors.

Features
Upload multiple PDF files at once.

Split and process large documents automatically.

Ask questions about the uploaded PDFs.

Get smart answers powered by LLAMA 3.

Maintain conversation history during the chat session.

Tools & Libraries Used
Streamlit: To create an interactive web app easily.

LangChain: To handle document loading, text splitting, embedding, and conversational logic.

FAISS: To store and retrieve document chunks efficiently.

Hugging Face Embeddings: To convert text into embeddings for semantic search.

Groq (LLAMA 3 API): To power the AI model for generating smart answers.

PyMuPDF (via LangChain): To load and read PDFs.

dotenv: To load environment variables securely.

How It Works
Upload one or more PDF files.

The app reads and splits the documents into manageable chunks.

Text chunks are converted into embeddings using Hugging Face models.

Embeddings are stored in a FAISS vector store.

Ask questions! The app uses the LLAMA 3 model to retrieve relevant information and answer.

Setup Instructions
Clone the repository:

bash
Copy code
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Set up your .env file:

ini
Copy code
GROQ_API_KEY=your_groq_api_key_here
Run the Streamlit app:

bash
Copy code
streamlit run app.py
Environment Variables
GROQ_API_KEY: Your API key from Groq to access LLAMA 3.

Notes
Ensure that your Groq API key has access to the "llama3-70b-8192" model.

Large PDFs might take a few seconds to process.


