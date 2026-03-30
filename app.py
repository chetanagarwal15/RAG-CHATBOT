import cohere
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

#INIT COHERE
PDF_PATH = r"C:\Users\hp\Desktop\RAG-Chatbot\sample.pdf"
COLLECTION_NAME = "pdf_docs"

co = cohere.Client("eECRY4vF4HdxUAyv8KfeJC74jpWXDMstnYBHjxgo") # <-- put your key

#Initialize Qdrant (local memory)
qdrant = QdrantClient(":memory:")

#STEP 1: LOAD PDF
def load_pdf(file_path):
    reader = PdfReader(file_path)

    text = ""
    for page in reader.pages:
        text += page.extract_text()

    return text

#STEP 2: CHUNKING
def chunk_text(text, chunk_size=500):
    chunks = []

    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        chunks.append(chunk)

    return chunks

#STEP 3: EMBEDDING
def embed_chunks(chunks):
    response = co.embed(
        texts=chunks,
        model="embed-english-v3.0",
        input_type="search_document"
    )
    return response.embeddings

#STEP 4: CREATE COLLECTION
def create_collection():
    if not qdrant.collection_exists(COLLECTION_NAME):
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=1024,
                distance=Distance.COSINE
            )
        )
#STEP 5: STORE IN QDRANT
def store_in_qdrant(chunks, embeddings):
    points = []

    for i in range(len(chunks)):
        points.append(
            PointStruct(
                id=i,
                vector=embeddings[i],
                payload={
                    "text": chunks[i]
                }
            )
        )

    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

def embed_query(query):
    response = co.embed(
        texts=[query],
        model="embed-english-v3.0",
        input_type="search_query"
    )
    return response.embeddings[0]

def search_qdrant(query):
    query_vector = embed_query(query)

    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=3 #top 3 chunks
    ).points

    return [r.payload["text"] for r in results]

def generate_answer(query, context_chunks):
    context = "\n\n".join(context_chunks)

    response = co.chat(
        model="command-a-03-2025",
        message=f"""
        Answer the question based ONLY on the context below.

        Context:
        {context}

        Question: {query}
        """
    )

    return response.text.strip()

# TEST
if __name__ == "__main__":
    print("Loading PDF...")
    text = load_pdf(PDF_PATH)

    print("Chunking text...")
    chunks = chunk_text(text)

    print("Creating embeddings...")
    embeddings = embed_chunks(chunks)

    print("Creating Qdrant collection...")
    create_collection()

    print("Storing data un Qdrant...")
    store_in_qdrant(chunks, embeddings)

    print("\n✅ RAG system ready!")

    while True:

        query = input("\nAsk A Question (or type 'exit'): ")
        
        if query.lower() == "exit":
            break

        #Retrieve relevant chunks
        context_chunks = search_qdrant(query)

        #Generate answer
        answer = generate_answer(query, context_chunks)

        print("\n🤖 Answer:\n")
        print(answer)