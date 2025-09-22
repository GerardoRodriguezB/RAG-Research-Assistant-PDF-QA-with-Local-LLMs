# RAG-Research-Assistant-PDF-Q&A-with-Local-LLMs 

This repository implements a Retrieval-Augmented Generation (RAG) system for answering questions from research papers. It combines a dual-model architecture: all-MiniLM-L6-v2 for semantic search and Meta's llama3.2 (via Ollama) for response generation. The embedding model converts PDF content into vector embeddings to enable efficient semantic search, while the generative model synthesizes accurate answers from retrieved context. The system works fully offline, making it suitable for privacy-sensitive research.

## Environment Setup

Create an Anaconda environment using `python=3.10`. Navigate to the root folder of the project and install requirements

```bash
pip install -r requeriments.txt
```

## Install PyTORCH

If your machine has a CUDA compatible GPU, install:

```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
```

Otherwise, install the versions for CPU

```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
```

## Install OLLAMA

1. **Install Ollama**: Download from [ollama.ai](https://ollama.ai/) and follow installation instructions
2. **Pull Model**: Run `ollama pull llama3.2` in your terminal to download the language model

## Folder with the PDF Files

Create a folder named papers and place your PDF files inside. The following directory tree shows the expected project structure:

```
root_directory/
├── papers/                # Folder with your PDF files
├── vector_db/             # Folder where embeddings and indexes will be stored
│   RAG_load.ipynb         # Script to process PDFs and build the vector DB
│   RAG_assistant.ipynb    # Script to query the assistant
```

## Usage

Run the file `RAG_load.ipynb` to process the PDF files the first time, and then every time you want to use the assistant use `RAG-assistant.ipynb`.



## Example of a Query

Here we show the output of the model when we ask for "Noticias de cine"

<img src="News.png" alt="News" width="900" />











