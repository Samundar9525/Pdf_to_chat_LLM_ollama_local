# PDF-to-Chatbot with Local GGUF Model

This project lets you chat with a PDF using a local GGUF LLM model, embeddings, and Gradio UI.  
The model is downloaded from Google Drive at runtime, so you don’t need to upload large files to GitHub.

---

## 🚀 Features

- Extracts and chunks text from a fixed PDF.
- Uses sentence-transformers for embeddings and Chroma for retrieval.
- Loads a quantized GGUF model (e.g., Mistral) via ctransformers.
- Gradio chatbot interface for Q&A.

---
## 📸 Snapshots

Here is what the app looks like:

![Chatbot UI](![image](https://github.com/user-attachments/assets/ef3290a5-a445-4b72-b6d3-ba03b866b68b) ![Chatbot UI](![image](![{4AA556E5-6D89-4DBD-A5D7-73ECED3AB25E}](https://github.com/user-attachments/assets/d6a06de3-73dc-417c-941c-fe0e1c7dc405)
![Chatbot UI](![image](https://github.com/user-attachments/assets/fd886051-bf16-4367-bfe7-781458d94a4d)
![Chatbot UI](![image](![image](https://github.com/user-attachments/assets/e8ef5814-c371-40a9-81c8-61f0d283eb40)
## 📝 Project Structure

```
.
├── app.py                # Main application code
├── requirements.txt      # Python dependencies
├── 995.pdf               # Your PDF file (or update filename in code)
└── .gitignore            # Ignores large model files
```

---

## 🛠️ Setup & Deployment on Hugging Face Spaces

### 1. **Prepare Your Files**

- Place your code (e.g., `app.py`), `requirements.txt`, and your PDF (e.g., `995.pdf`) in your project folder.
- **Do NOT upload your model `.gguf` or `.zip` to GitHub.**  
  The model will be downloaded from Google Drive at runtime.

### 2. **Create a `.gitignore`**

Add the following to `.gitignore` to avoid uploading large model files:

```
*.gguf
*.zip
model/
```

### 3. **requirements.txt Example**

```
gradio
langchain
langchain-huggingface
ctransformers
pdfplumber
gdown
chromadb
sentence-transformers
```

### 4. **Download the required model locally that need to be used**

### 5. **Create a Hugging Face Space**

- Go to [Hugging Face Spaces](https://huggingface.co/spaces).
- Click **"Create new Space"**.
- Choose **Gradio** as the SDK.
- Connect your GitHub repo or upload your files directly.

### 6. **Configure Your Model Download in `app.py`**

Make sure your code includes logic to download and unzip the model from Google Drive, e.g.:

```python
import os, zipfile

MODEL_ZIP_URL = "https://drive.google.com/uc?id=YOUR_FILE_ID"
MODEL_ZIP_PATH = "mistral-7b-instruct-v0.1.Q2_K.zip"
MODEL_PATH = "mistral-7b-instruct-v0.1.Q2_K.gguf"

def download_and_unzip_model():
    if not os.path.exists(MODEL_PATH):
        if not os.path.exists(MODEL_ZIP_PATH):
            import gdown
            gdown.download(MODEL_ZIP_URL, MODEL_ZIP_PATH, quiet=False)
        with zipfile.ZipFile(MODEL_ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(".")
download_and_unzip_model()
```

**Replace `YOUR_FILE_ID` with your actual Google Drive file ID.**

### 7. **Wait for Build & Test**

- Hugging Face Spaces will install dependencies and run your app.
- If you see errors about missing files, check your model path and the output of `os.listdir(".")` in your logs.

---

## 🧑‍💻 Usage

- Open your Space.
- Ask questions about the PDF in the Gradio chat interface.

---

## 📝 Notes

- The model is **not stored in the repo**; it is downloaded at runtime.
- If your PDF filename is different, update it in the code.
- For large models, initial startup may take several minutes.

---

## 📄 License

MIT

---

## 🤝 Credits

- [LangChain](https://github.com/langchain-ai/langchain)
- [ctransformers](https://github.com/marella/ctransformers)
- [Gradio](https://github.com/gradio-app/gradio)
