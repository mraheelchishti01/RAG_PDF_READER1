# Free PDF RAG Lab (Streamlit)

A beginner-friendly RAG teaching demo:
1. Upload a text-based PDF.
2. Extract PDF text and split it into overlapping chunks.
3. Retrieve relevant chunks using TF-IDF cosine similarity.
4. Send retrieved context to a Hugging Face hosted instruction model.
5. Display the answer and retrieved passages/page numbers.

## Files
- `app.py` — Streamlit interface and chat flow
- `rag.py` — PDF parsing, chunking, retrieval, prompt construction
- `requirements.txt` — dependencies

## Run locally
Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

In the app sidebar, paste a Hugging Face access token. Create one at:
https://huggingface.co/settings/tokens

Use a fine-grained token with the minimum permissions needed. Do not commit the token to GitHub or put it in source code.

## Deploy to Streamlit Community Cloud
1. Create/sign in to GitHub.
2. Create a public repository, e.g. `free-pdf-rag-lab`.
3. Upload `app.py`, `rag.py`, `requirements.txt`, and this `README.md` to the repository root.
4. Visit https://share.streamlit.io/ and sign in with GitHub.
5. Select **Create app**, choose the repository, branch (usually `main`), and main file path `app.py`.
6. Deploy. When the app opens, enter your Hugging Face token in the sidebar and upload a PDF.
7. If deployment fails, open the app's **Manage app → Logs** and check the missing dependency or model error.

## Important free-tier limitations
- Hugging Face hosted inference is not guaranteed to be unlimited or always available. Free usage is subject to provider/model rate limits, availability, and policy changes.
- The model ID shown is a default example. If it is unavailable to your account, choose an instruction/chat model currently supported by Hugging Face Inference Providers and update the Model ID field.
- This simple version uses TF-IDF, not a vector database or neural embeddings. It is intentionally lightweight for a classroom lab.
- Scanned/image-only PDFs need OCR; this version expects selectable text.
- Uploaded files and chat history are held in the running Streamlit session. Do not upload confidential documents to a public demo.
- For a fully local model without hosted inference limits, install Ollama and run a local model, but that local model will not run inside Streamlit Community Cloud by itself.
