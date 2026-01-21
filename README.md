# Cook Compass

**Cook Compass** is a RAG-based (Retrieval-Augmented Generation) recipe assistant designed to solve the "dinner
dilemma." Unlike generic chatbots that often hallucinate ingredients, Cook Compass retrieves verified recipes from a
structured dataset based on the ingredients you actually have at home.

## Live Demo

If you don't want to set up the environment locally, you can test the deployed version here:
**[https://cook-compass.streamlit.app/](https://cook-compass.streamlit.app/)**

## Technical Report

For a detailed analysis of our architecture, user journey, and evaluation results (Abstract, Description, Evaluation,
Reflection), please refer to the PDF version of our report or view the markdown source here: *
*[docs/technical_report.md](docs/technical_report.md)**.

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
the system checks if the ChromaDB vector store is empty. If it is, it automatically runs the ingestion pipeline to
process the CSV and create embeddings.

---

## Project Setup

Here is a quick overview of the files to help you navigate the code:

```text
cook-compass/
├── app.py                      # Main entry point. Runs the Streamlit UI.
├── config.yaml                 # Settings for models, paths, and file names.
├── .env                        # Store your API_TOKEN here (not committed).
├── requirements.txt            # Python dependencies.
├── docs/                       # Contains the technical report and the project plan.
├── data/
│   └── small_recipes.csv       # The source dataset.
├── db/                         # Created automatically. Stores the ChromaDB vectors.
├── notebooks/
│   ├── Create_small_dataset.ipynb           # Preprocessing script to create the recipe subset.
│   ├── generate_test_queries.ipynb          # Generates synthetic user queries for testing.
│   └── LLM_as_judge+visualise_results.ipynb # Runs the evaluation pipeline and visualizes results.
└── src/
    ├── ingest.py               # Logic to load CSV, clean data, and save to ChromaDB.
    ├── inference.py            # The RAG engine. Handles retrieval and LLM generation.
    ├── retrievers.py           # Haystack pipeline definitions (Embeddings + Retrieval).
    ├── prompts.py              # System prompts for the AI Chef persona.
    ├── keywords.py             # Defines the list of dietary keywords (e.g., vegan, keto) for filtering.
    └── utils.py                # Helper functions for paths and config loading.
```

## Tech Stack

* Framework: Haystack 2.0
* Frontend: Streamlit
* Vector Database: ChromaDB
* Embeddings: BAAI/bge-base-en-v1.5
* LLM: Qwen/Qwen2.5-7B-Instruct (via Hugging Face)

## Evaluation

We evaluate the system using an **"LLM-as-a-Judge"** approach. The notebook
`notebooks/LLM_as_judge+visualise_results.ipynb` uses a stronger model to grade responses on a 0-10 scale:

1. **Relevance:** Does the recipe match the ingredients and constraints?
2. **Healthiness:** Nutritional assessment (balance of ingredients).
3. **Taste:** Culinary logic and flavor combinations.

**Results:** The system achieves high scores for relevance and taste (~8/10) by retrieving from a curated dataset rather
than generating recipes from scratch. See `docs/technical_report.pdf` for more results.