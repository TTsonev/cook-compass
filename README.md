# Cook Compass

Cook Compass is a RAG-based (Retrieval-Augmented Generation) recipe assistant. It focuses on finding and recommending
vegetarian recipes based on user ingredients. It uses a vector database to search a subset of Food.com data and an LLM
to generate helpful responses.

## Live Demo

If you don't want to set up the environment locally, you can test the deployed version here:
**[https://cook-compass.streamlit.app/](https://cook-compass.streamlit.app/)**

---

## Local Setup

If you want to run the project locally, follow these steps.

### 1. Prerequisites

* Python (at least on 3.12)
* A **Hugging Face API Token** (Read permission is enough).

### 2. Installation

Clone the repository and install the dependencies:

```bash
pip install -r requirements.txt
```

### 3. Configuration

1. Create a `.env` file in the root directory.
2. Add your Hugging Face token:
   ```env
   API_TOKEN=hf_your_token_here
   ```
3. **Dataset:** Make sure you have the recipe dataset file (CSV) inside a `data/` folder. The system looks for the file
   specified in `config.yaml` (default: `small_recipes.csv`).

### 4. Running the App

Start the Streamlit interface:

```bash
streamlit run app.py
```

**Note on Database Ingestion:** You don't need to run a separate script to build the database. When you run `app.py`,
the
system checks if the ChromaDB vector store is empty. If it is, it automatically runs the ingestion pipeline to process
the CSV and create embeddings.

---

## Project Setup

Here is a quick overview of the files to help you navigate the code:

```text
cook-compass/
├── app.py                  # Main entry point. Runs the Streamlit UI.
├── config.yaml             # Settings for models, paths, and file names.
├── .env                    # Store your API_TOKEN here (not committed).
├── requirements.txt        # Python dependencies.
├── LLM_as_judge.ipynb      # Notebook for evaluating response quality.
├── docs/                   # Contains the technical report and the project plan.
├── data/
│   └── small_recipes.csv   # The source dataset.
├── db/                     # Created automatically. Stores the ChromaDB vectors.
└── src/
    ├── ingest.py           # Logic to load CSV, clean data, and save to ChromaDB.
    ├── inference.py        # The RAG engine. Handles retrieval and LLM generation.
    ├── retrievers.py       # Haystack pipeline definitions (Embeddings + Retrieval).
    ├── prompts.py          # System prompts for the AI Chef persona.
    └── utils.py            # Helper functions for paths and config loading.
```

## Evaluation

We evaluate the system using an "LLM-as-a-Judge" approach. The `LLM_as_judge.ipynb` notebook uses a stronger model (like
Llama-3) to score the recommendations based on:

1) Relevance (Does it match the user's ingredients?)
2) Healthiness (Nutritional balance)
3) Taste (Culinary logic)

## Tech Stack

* Framework: Haystack 2.0
* Frontend: Streamlit
* Vector Database: ChromaDB
* Embeddings: BAAI/bge-base-en-v1.5
* LLM: Qwen/Qwen3-VL-8B-Instruct (via Hugging Face Serverless API)