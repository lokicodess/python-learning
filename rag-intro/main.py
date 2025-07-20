import os
import PyPDF2
import numpy as np
import faiss
import pickle
import gradio as gr
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai

# ---------- Configure Gemini API ----------
genai.configure(api_key="")  # Replace with your actual API key

# ---------- Initialize Gemini Model ----------
model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")


# ---------- Extract text from PDF ----------
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


# ---------- Split text into chunks ----------
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(text)
    return chunks


# ---------- Get Gemini embeddings ----------
def get_gemini_embeddings(texts):
    embeddings = []
    for text in texts:
        response = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_document"
        )
        embedding_vector = response.get("embedding") or response["embedding"]
        embeddings.append(embedding_vector)
    return embeddings


# ---------- Build FAISS index ----------
def build_faiss_index(vectors):
    dim = len(vectors[0])
    index = faiss.IndexFlatL2(dim)
    np_vectors = np.array(vectors).astype("float32")
    index.add(np_vectors)
    return index


# ---------- Generate response from Gemini ----------
def generate_gemini_text(prompt):
    response = model.generate_content(prompt)
    return response.text


# ---------- Query RAG pipeline ----------
def query_gemini_rag(query, index, chunks, top_k=3):
    query_vector = get_gemini_embeddings([query])[0]
    query_vector_np = np.array([query_vector]).astype("float32")
    
    D, I = index.search(query_vector_np, top_k)
    relevant_chunks = [chunks[i] for i in I[0]]

    context = "\n\n".join(relevant_chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

    response = generate_gemini_text(prompt)
    return response


# ---------- Load or Build Cache ----------
PDF_PATH = "sample.pdf"
INDEX_PATH = "my_index.faiss"
CHUNKS_PATH = "chunks.pkl"

if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
    print("🔄 Loading cached index and chunks...")
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
else:
    print("⚙️  Building index and chunks from scratch...")
    pdf_text = extract_text_from_pdf(PDF_PATH)
    chunks = split_text(pdf_text)
    embeddings = get_gemini_embeddings(chunks)
    index = build_faiss_index(embeddings)

    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)


# ---------- Gradio Interface ----------
def handle_query(user_query):
    return query_gemini_rag(user_query, index, chunks)

interface = gr.Interface(
    fn=handle_query,
    inputs=gr.Textbox(lines=2, placeholder="Ask something about the PDF..."),
    outputs="text",
    title="Gemini RAG PDF QA",
    description="Ask questions based on the PDF content using Gemini + FAISS"
)

interface.launch()
