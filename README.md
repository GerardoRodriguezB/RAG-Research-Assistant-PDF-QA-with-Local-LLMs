# RAG-Research-Assistant-Ussing-LLM

This repository implements a Retrieval-Augmented Generation (RAG) system for research paper Q&A using a dual-model architecture. The pipeline employs `all-MiniLM-L6-v2` for semantic text representation and Meta's `llama3.2` for response generation. The embedding model converts PDF content into vector embeddings to enable efficient semantic search, while the generative model synthesizes accurate answers from retrieved context. A key advantage of this assistant is its complete offline operation using Ollama and local models.


## Environment Setup

Create an Anaconda environment using `python=3.10`. Navigate to the root folder of the project and install requeriments

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

In the following scheme it is shown a tree of folders. Create a folder named `papers` and put the PDF files inside.

```
root_directory/
├── papers/
├── vector_db/
│   RAG_load.ipynb
│   RAG_assistant.ipynb
```

## Usage

Run the file `RAG_load.ipynb` to process the PDF files the first time, and then every time you wantr to use the assistant use `RAG-assistant.ipynb`.



## Example of a Query

Here we show the output of the model when we ask for "Noticias de cine"

<img src="News.png" alt="News" width="900" />











