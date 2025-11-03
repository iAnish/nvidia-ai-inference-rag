# ==============================
# enable RAG with inference using the Llama-3.2-1B-Instruct using Pytorhc. We will use streamlit framework to create the UI and provide ioption to upload PDF for RAG
# ==============================

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer, util
import PyPDF2
from io import BytesIO

# Constants
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"  # Use the instruct version for better prompt handling
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Lightweight embedding model

# Load models
@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if device == "cuda" else -1)
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return generator, embedder, device

generator, embedder, device = load_models()

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

# Function to split text into chunks
def split_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

# Streamlit UI
st.title("Llama 1B Inference with RAG")

# Sidebar for configurations
st.sidebar.header("Inference Configurations")
temperature = st.sidebar.slider("Temperature", min_value=0.1, max_value=2.0, value=0.7, step=0.1)
top_k = st.sidebar.slider("Top K", min_value=1, max_value=100, value=50)
top_p = st.sidebar.slider("Top P", min_value=0.1, max_value=1.0, value=0.95, step=0.05)
max_length = st.sidebar.slider("Max Length", min_value=50, max_value=1000, value=200, step=50)
repetition_penalty = st.sidebar.slider("Repetition Penalty", min_value=1.0, max_value=2.0, value=1.2, step=0.1)

# File uploader for PDF
uploaded_pdf = st.file_uploader("Upload PDF for RAG (optional)", type="pdf")

# Process PDF if uploaded
document_chunks = None
chunk_embeddings = None
if uploaded_pdf:
    with st.spinner("Processing PDF..."):
        pdf_text = extract_text_from_pdf(BytesIO(uploaded_pdf.read()))
        document_chunks = split_text(pdf_text)
        chunk_embeddings = embedder.encode(document_chunks, convert_to_tensor=True).to(device)
    st.success("PDF processed successfully!")

# Query input
query = st.text_input("Enter your query:")

# Generate button
if st.button("Generate Response"):
    if query:
        with st.spinner("Generating response..."):
            # Prepare prompt
            if document_chunks and chunk_embeddings is not None:
                # RAG: Retrieve top chunks
                query_embedding = embedder.encode(query, convert_to_tensor=True).to(device)
                cos_scores = util.pytorch_cos_sim(query_embedding, chunk_embeddings)[0]
                top_k_retrieve = min(3, len(document_chunks))  # Retrieve top 3 chunks
                top_results = torch.topk(cos_scores, k=top_k_retrieve)
                retrieved = "\n\n".join([document_chunks[idx] for idx in top_results[1]])
                prompt = f"Context:\n{retrieved}\n\nQuery: {query}\nAnswer:"
            else:
                prompt = f"Query: {query}\nAnswer:"

            # Generate
            response = generator(
                prompt,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                num_return_sequences=1,
                do_sample=True
            )[0]['generated_text']

            # Display response (remove the prompt part)
            generated_text = response[len(prompt):].strip()
            st.subheader("Generated Response:")
            st.write(generated_text)
    else:
        st.warning("Please enter a query.")
