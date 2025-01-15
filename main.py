import os
import streamlit as st
import pickle
import langchain
import time
from langchain_openai import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables (e.g., API keys) from a .env file
load_dotenv()

# Set up the Streamlit app with a title
st.title("News Insights QA Tool 📈")

# Sidebar for inputting URLs
st.sidebar.title("News Article URLs")

# Initialize the OpenAI LLM with specific parameters
llm = OpenAI(temperature=0.9, max_tokens=500)

# Collect up to three URLs from the user via the sidebar
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

# Button in the sidebar to trigger URL processing
process_url_clicked = st.sidebar.button("Process URLs")

# Placeholder to store the vectorstore object
vectorstore_openai = None

# Placeholder for showing intermediate status updates in the app
main_placeholder = st.empty()

# If the "Process URLs" button is clicked
if process_url_clicked:
    # Load data from the provided URLs using UnstructuredURLLoader
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...✅")
    data = loader.load()

    # Split the loaded data into smaller chunks with overlap for context
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", ","],  # Define separators for splitting
        chunk_size=1000,  # Maximum size of each chunk
        chunk_overlap=200  # Overlap between chunks
    )
    main_placeholder.text("Text Splitter Started...✅")
    docs = text_splitter.split_documents(data)  # Split the data into chunks

    # Generate embeddings for the chunks and store them in a FAISS vector database
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅")
    time.sleep(2)

    # Save the FAISS index to local storage for future use
    vectorstore_openai.save_local("faiss_index")
    time.sleep(2)

# Function to load the FAISS index from local storage (cached for performance)
def load_faiss_index():
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Check if the FAISS index exists and load it if available
if os.path.exists("faiss_index"):
    vectorstore_openai = load_faiss_index()

# Ensure that query processing is only enabled if the FAISS index is loaded
if vectorstore_openai:
    # Text input for the user's question
    query = main_placeholder.text_input("Enter your question:")
    if query.strip():  # Check if the query is not empty
        # Set up the RetrievalQAWithSourcesChain to process the query
        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm,  # Use the initialized LLM
            retriever=vectorstore_openai.as_retriever()  # Use the vectorstore retriever
        )
        result = chain({"question": query}, return_only_outputs=True)

        # Display the generated answer
        st.header("Answer")
        st.write(result["answer"])

        # Display the sources for the answer, if available
        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")  # Split sources into a list
            for source in sources_list:
                st.write(source)
    else:
        # Warn the user if the query is empty
        st.warning("Please enter a valid question.")
else:
    # Inform the user to process URLs first before querying
    st.info("Please process URLs first by entering them in the sidebar and clicking 'Process URLs'.")
