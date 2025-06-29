from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import gradio as gr
from ctransformers import AutoModelForCausalLM
import pdfplumber
import os
import zipfile

# Download zipped model from Google Drive if not present
MODEL_ZIP_URL = "https://drive.google.com/uc?id=1N8C5YEQdSPkmssE5kbWwLJ__H5nZRtSd"  # <-- Your zip file's Google Drive ID
MODEL_ZIP_PATH = "mistral-7b-instruct-v0.1.Q2_K.zip"
MODEL_PATH = "model/model/mistral-7b-instruct-v0.1.Q2_K.gguf"

def download_and_unzip_model():
    if not os.path.exists(MODEL_PATH):
        if not os.path.exists(MODEL_ZIP_PATH):
            import gdown
            print("Downloading zipped model from Google Drive...")
            gdown.download(MODEL_ZIP_URL, MODEL_ZIP_PATH, quiet=False)
            print("Model zip downloaded.")
        print("Unzipping model...")
        with zipfile.ZipFile(MODEL_ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(".")
        print("Model unzipped.")

download_and_unzip_model()

# 1. Load and chunk the PDF (no upload, use fixed file)
print("Loading PDF...")
with pdfplumber.open("995 Large Wheel Loader _ Cat _ Caterpillar.pdf") as pdf:
    text = ""
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
print("Loaded PDF.")

print("Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(text)
print(f"Created {len(chunks)} chunks.")

# 2. Create embeddings and vector store for retrieval
print("Creating embeddings and vector store...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma.from_texts(chunks, embedding=embeddings)
retriever = db.as_retriever()
print("Vector store and retriever ready.")

# 3. Load your local GGUF model with ctransformers
print("Loading local GGUF model...")
assert os.path.exists(MODEL_PATH), f"Model file not found at {MODEL_PATH}"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    model_type="mistral",
    max_new_tokens=256,
    gpu_layers=0,
    local_files_only=True,
    hf=False,  # Prevents Hugging Face Hub logic
)
print("Model loaded.")

# 4. Define the chat function
def chat_fn(question):
    docs = retriever.get_relevant_documents(question)
    context = "\n".join([doc.page_content for doc in docs[:2]])
    prompt = (
        f"Use only the following context to answer the question. "
        f"If the answer is not in the context, say 'I don't know.'\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    response = model(prompt)
    return response

# 5. Launch Gradio chatbot
print("Launching Gradio chatbot...")
gr.Interface(
    fn=chat_fn,
    inputs="text",
    outputs="text",
    title="PDF Q&A Chatbot (Local GGUF Model)"
).launch()

print("Files/folders in current directory:", os.listdir("."))
if os.path.exists("model"):
    print("Files in 'model':", os.listdir("model"))
    if os.path.exists("model/model"):
        print("Files in 'model/model':", os.listdir("model/model"))