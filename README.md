# 📚 PaperMind AI – Multimodal RAG System

A **Multimodal Retrieval-Augmented Generation (RAG)** system that allows users to ask questions about 30 research papers using text and images.

PaperMind AI retrieves relevant text chunks and research figures, then uses Groq LLMs to generate context-based answers with source information.

## ✨ Features

- **Text RAG:** Retrieve relevant text chunks from 30 research papers.
- **Image RAG:** Retrieve relevant research figures and images using CLIP embeddings.
- **Query Router:** Automatically routes questions to the text or image retrieval pipeline based on keywords.
- **Vision LLM:** Analyzes retrieved figures and explains their content.
- **Chat History:** Maintains previous conversations in the Streamlit sidebar.
- **Source Information:** Displays relevant paper names, page numbers, and image sources when metadata is available.
- **Paper Filtering:** Filter retrieval results using paper metadata.

## 🏗️ Architecture

### Text Retrieval Pipeline

User Question  
↓  
Query Router  
↓  
Text Query  
↓  
Sentence Transformer Embeddings  
↓  
ChromaDB Text Collection (`paper_30`)  
↓  
Retrieve Relevant Text Chunks  
↓  
Groq LLM (`openai/gpt-oss-20b`)  
↓  
Answer + Source Information

### Image Retrieval Pipeline

User Question  
↓  
Query Router  
↓  
Image Query  
↓  
CLIP Text Embeddings  
↓  
ChromaDB Image Collection (`paper_30_images`)  
↓  
Retrieve Top Matching Images  
↓  
Vision LLM  
↓  
Image Explanation + Source Information

## 📊 Dataset Information

The system uses a collection of 30 research papers.

| Component | Information |
|---|---|
| Research Papers | 30 |
| Text Chunks | 3,756 |
| Images / Figures | 1,411 |
| Text Collection | `paper_30` |
| Image Collection | `paper_30_images` |

## 🧠 Models Used

| Purpose | Model |
|---|---|
| Text Embeddings | `all-MiniLM-L6-v2` |
| Image Embeddings | `clip-ViT-B-32` |
| Text Generation | `openai/gpt-oss-20b` |
| Image Understanding | Vision model configured in `app.py` |

**Embedding dimensions:**
- Text embeddings: 384 dimensions
- CLIP image embeddings: 512 dimensions

## 🛠️ Tech Stack

- **Programming Language:** Python
- **Frontend:** Streamlit
- **Backend:** FastAPI
- **Vector Database:** ChromaDB
- **Text Embeddings:** Sentence Transformers
- **Image Retrieval:** CLIP (ViT-B-32)
- **LLM API:** Groq
- **Image Processing:** Pillow
- **Deep Learning:** PyTorch

## 💻 Hardware & Performance

- Tested on NVIDIA GTX 1650 (4GB VRAM).
- Sentence Transformer and CLIP models are configured to run on CPU using `device="cpu"`.
- CPU execution avoids the CUDA-related errors encountered in the development environment.

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd RAG_with_Langchain
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Groq API Key

Create a folder named `.streamlit` in the project directory.

Inside it, create `secrets.toml`:

```toml
GROQ_API_KEY = "YOUR_GROQ_API_KEY"
```

The application reads the API key using:

```python
groq_client = Groq(
    api_key=st.secrets["GROQ_API_KEY"]
)
```

**Important:** Never upload `secrets.toml` or your API key to GitHub.

Add this to `.gitignore`:

```gitignore
.streamlit/secrets.toml
```

### 4. Save Models Locally (Optional)

```bash
python save_models.py
```

This saves the embedding models locally in:

```text
models/
├── text_model/
└── clip_model/
```

Run this only if your `save_models.py` script is configured to save these models and your application loads them from these locations.

### 5. Run the Streamlit Application

```bash
streamlit run app.py
```

Open the application:

```text
http://localhost:8501
```

### 6. Run FastAPI Backend (Optional)

```bash
uvicorn main:app --reload --port 8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

## 📁 Project Structure

```text
RAG_with_Langchain/
│
├── app.py                  # Streamlit user interface
├── main.py                 # FastAPI backend
├── save_models.py          # Local model saving script
├── requirements.txt        # Python dependencies
├── chroma_db/              # Persistent ChromaDB storage
│
├── Data/
│   └── images/             # Research paper images and figures
│
├── models/
│   ├── text_model/         # Saved text embedding model
│   └── clip_model/         # Saved CLIP model
│
└── .streamlit/
    └── secrets.toml        # Groq API key (do not upload)
```

## ⚠️ Known Limitations

- **CLIP Image Retrieval:** CLIP may retrieve visually similar but contextually incorrect charts. It is not specifically trained to understand every research chart or diagram.
- **Query Routing:** The current router uses keyword-based detection. Some image-related questions may be routed to the text pipeline.
- **Vision Model Availability:** Groq model availability can change. Use a currently supported vision model and keep its model ID updated in `app.py`.
- **CPU Performance:** Running embedding models on CPU can increase retrieval time.
- **Image Paths:** Images must be available in `Data/images/` and their stored filenames must match the ChromaDB records.
- **API Dependency:** Answer generation and image understanding require a working Groq API key and supported models.

## 🔮 Future Improvements

- Add OCR and caption-based retrieval for research charts.
- Improve image retrieval accuracy using better reranking techniques.
- Implement hybrid search using BM25 and dense retrieval.
- Improve query routing using an LLM-based router.
- Explore Qdrant for scalable vector storage.
- Add more research papers and improve evaluation.

## 👤 Author

Built as a portfolio project to explore **Retrieval-Augmented Generation (RAG), Vector Databases, Text Embeddings, Image Retrieval, and Multimodal AI**.

**Aakash S. Jadhav**
