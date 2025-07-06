# Medicare & You PDF QA API

This project is a FastAPI-based question-answering (QA) system over the official Medicare & You PDF document. It uses semantic search, embeddings, and OpenRouter's Mistral model to answer user queries with references to the source page and confidence scores.

## Features
- Extracts and chunks text from a Medicare PDF.
- Stores semantic chunks in a ChromaDB vector database.
- Embeds queries and retrieves the most relevant chunks.
- Calls OpenRouter's Mistral model to generate structured JSON answers.
- Returns answers with source page, confidence score, and chunk size.
- Simple HTML UI and REST API endpoint.

## Requirements
- Python 3.10+
- [OpenRouter API key](https://openrouter.ai/)

## Setup

1. **Clone the repository**

2. **Install dependencies**

```bash
python -m venv myenv
myenv\Scripts\activate  # On Windows
pip install -r requirements.txt
```

3. **Add your OpenRouter API key**

Create a `.env` file in the project root:
```
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

4. **Add the Medicare PDF**

Place your `10050-medicare-and-you.pdf` (or the correct PDF) in the project root.

5. **Run the server**

```bash
python main.py
```

The API will be available at [http://127.0.0.1:8000](http://127.0.0.1:8000)

## Usage

- Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser for the UI.
- Or POST a JSON query to `/query`:

```json
{
  "query": "When does Medicare enrollment begin?"
}
```

- The response will be a structured JSON object:

```json
{
  "answer": "Medicare enrollment begins on October 15 and ends on December 7 each year...",
  "source_page": 15,
  "confidence_score": 0.92,
  "chunk_size": 230
}
```

## Notes
- The first run will process and embed the PDF; subsequent runs are faster.
- Make sure your OpenRouter API key is valid and has access to the Mistral model.

## License
MIT
