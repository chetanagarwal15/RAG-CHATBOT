import os
import streamlit as st
import cohere
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# API KEY
co = cohere.Client(os.getenv("COHERE_API_KEY"))

# Qdrant
qdrant = QdrantClient(":memory:")
COLLECTION_NAME = "pdf_docs"

# Functions
def load_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def chunk_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def embed_chunks(chunks):
    return co.embed(
        texts=chunks,
        model="embed-english-v3.0",
        input_type="search_document"
    ).embeddings

def embed_query(query):
    return co.embed(
        texts=[query],
        model="embed-english-v3.0",
        input_type="search_query"
    ).embeddings[0]

def create_collection():
    if not qdrant.collection_exists(COLLECTION_NAME):
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
        )

def store(chunks, embeddings):
    points = []
    for i in range(len(chunks)):
        points.append(PointStruct(id=i, vector=embeddings[i], payload={"text": chunks[i]}))
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

def search(query):
    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=embed_query(query),
        limit=3
    ).points
    return [r.payload["text"] for r in results]

def generate_answer(query, context):
    context = "\n\n".join(context)
    response = co.chat(
        model="command-a-03-2025",
        message=f"""
        Answer using context if possible, else use your knowledge.

        Context:
        {context}

        Question: {query}
        """
    )
    return response.text

# UI
st.title("📄 RAG Chatbot")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    text = load_pdf(uploaded_file)
    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks)

    create_collection()
    store(chunks, embeddings)

    st.success("PDF processed!")

    query = st.text_input("Ask a question")

    if query:
        context = search(query)
        answer = generate_answer(query, context)
        st.write(answer)