# Retrieval-Augmented Generation (RAG) Streamlit App

# Still need to figure out how to install required libraries
# Uncomment and run the below command ***
# !pip install streamlit faiss-cpu transformers sentence-transformers

import numpy as np 
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import fitz 

# Load embedding model and generator model
@st.cache_resource
def loadModel():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    generator = pipeline("text2text-generation", model="google/flan-t5-small")
    return embedder, generator

embedder, generator = loadModel()

# Function to read PDF text
def load_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Chunking test
def chunk(text, chunk_size=500, overlap=100):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i+chunk_size])
    return chunks

# Embed and index documents
@st.cache_resource
def create_index(chunks):
    doc_embeddings = embedder.encode(chunks, convert_to_numpy=True)
    index = faiss.IndexFlatL2(doc_embeddings.shape[1])
    index.add(doc_embeddings)
    return index, chunks

# Retrieve top-k relevant documents
def retrieve(query, index, chunks, k=2):
    query_embedding = embedder.encode([query])
    distances, indices = index.search(query_embedding, k)
    return [chunks[i] for i in indices[0]]

# Streamlit UI
st.title("📄 Retrieval-Augmented Generation (RAG) with Your Documents")

uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing uploaded file..."):
        full_text = load_pdf(uploaded_file)
        chunks = chunk(full_text)
        index, chunk_list = create_index(chunks)

    query = st.text_input("Enter your question:", "Where is the Eiffel Tower?")

    if query:
        retrieved_docs = retrieve(query, index, chunk_list)
        context = " ".join(retrieved_docs)
        input_text = f"Answer the question based on the context:\nContext: {context}\nQuestion: {query}"
        response = generator(input_text, max_new_tokens=100)

        st.subheader("Retrieved Chunks")
        for doc in retrieved_docs:
            st.write("-", doc)

        st.subheader("Generated Answer")
        st.write(response[0]['generated_text'])
else:
    st.info("Upload a PDF document to begin.")
