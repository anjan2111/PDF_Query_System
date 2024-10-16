from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import PyPDF2
import tempfile
from fastapi.middleware.cors import CORSMiddleware
import re
from sentence_transformers import SentenceTransformer, util
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import chromadb
from chromadb.utils import embedding_functions
import os
import uvicorn

app = FastAPI()
origins = [
    "http://localhost",
    "http://localhost:5500",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load models for embeddings and question generation
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
t5_model = T5ForConditionalGeneration.from_pretrained('t5-base')
t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')

# Initialize ChromaDB for embeddings storage
chroma_client = chromadb.Client()
chroma_collection = chroma_client.create_collection("pdf_embeddings", embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction())

# Global variables to store PDF text and embeddings
pdf_text = ""
pdf_metadata = []
pdf_embeddings = None

# Function to clean and preprocess the extracted PDF text
def clean_text(text):
    text = re.sub(r'\n+', '\n', text)  # Remove multiple newlines
    text = re.sub(r'[^\w\s\.\,]', '', text)  # Remove special characters (except punctuation)
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces and trim
    return text

# Function to extract text from the PDF file along with page numbers for citation
def extract_text_from_pdf(file_path):
    global pdf_text, pdf_metadata
    reader = PyPDF2.PdfReader(file_path)
    raw_text = ""
    pdf_metadata = []  # To store metadata like page numbers
    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text()
        raw_text += f"\n\nPage {page_num + 1}\n\n" + page_text
        # Store paragraph metadata with page number
        pdf_metadata.append({
            "page_number": page_num + 1,
            "text": clean_text(page_text)
        })
    pdf_text = clean_text(raw_text)

# Function to generate embeddings and store them in ChromaDB
def generate_embeddings_and_store(text):
    paragraphs = text.split("\n\n")  # Split the text into paragraphs
    embeddings = embedding_model.encode(paragraphs, convert_to_tensor=True)
    
    # Store each paragraph embedding in ChromaDB with metadata
    for i, paragraph in enumerate(paragraphs):
        chroma_collection.add(
            documents=[paragraph],
            metadatas={"paragraph_index": i},
            ids=[f"paragraph_{i}"]
        )

# Upload PDF endpoint
@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file.file.read())
            extract_text_from_pdf(temp_file.name)
        
        # Generate embeddings and store them in ChromaDB
        generate_embeddings_and_store(pdf_text)

        return {"message": "PDF uploaded, processed, and embeddings generated."}
    except Exception as e:
        return {"error": str(e)}

# Generate questions from PDF content using T5 model
def generate_questions_from_text(text, num_questions=3):
    input_text = f"generate questions: {text}"
    inputs = t5_tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True)
    outputs = t5_model.generate(inputs, max_length=150, num_beams=5, early_stopping=True, num_return_sequences=num_questions)

    questions = [t5_tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return questions

# Endpoint for generating questions
@app.get("/suggest_questions/")
async def suggest_questions():
    questions = generate_questions_from_text(pdf_text)
    return {"questions": questions}

# Function to process a query using embeddings and ChromaDB
def process_query_with_embeddings(query):
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)

    # Perform similarity search in ChromaDB
    search_results = chroma_collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=1
    )

    # Get the most relevant paragraph and its citation
    best_paragraph = search_results['documents'][0][0]
    best_metadata = search_results['metadatas'][0][0]
    paragraph_index = int(best_metadata['paragraph_index'])

    # Find the corresponding page for citation from the stored metadata
    citation_page = None
    for metadata in pdf_metadata:
        if metadata["text"] in best_paragraph:
            citation_page = metadata["page_number"]
            break

    return best_paragraph, {'page': citation_page}

from pydantic import BaseModel

# Define a Pydantic model to expect JSON input for the query
class QueryRequest(BaseModel):
    query: str

# Query PDF endpoint with citation support
@app.post("/query_pdf/")
async def query_pdf(request: QueryRequest):
    try:
        query = request.query
        result, citation = process_query_with_embeddings(query)
        return {"query_result": result, "citation": citation}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)