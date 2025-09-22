# RAG-Research-Assistant-Ussing-LLM

This repository implements a Retrieval-Augmented Generation (RAG) system for research paper analysis using a dual-model architecture. The system employs `all-MiniLM-L6-v2` for semantic text representation and Meta's `llama3.2` for response generation. The embedding model converts PDF content into vector embeddings enabling efficient semantic search, while the generative model synthesizes accurate answers based on the retrieved context.

- **Local Execution**: Entire pipeline runs offline using Ollama and local models
- **Academic Focus**: Optimized for research paper analysis and academic Q&A

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

## Example of a Query

Here we show the output of the model when we ask for "Noticias de cine"

<img src="News.png" alt="News" width="900" />










```
root_directory/
├── papers/
├── vector_db/
│   RAG_load.ipynb
│   RAG_assistant.ipynb
```
