# Free PDF RAG Lab (Streamlit)

A beginner-friendly RAG teaching demo:
1. Upload a text-based PDF.
2. Extract PDF text and split it into overlapping chunks.
3. Retrieve relevant chunks using TF-IDF cosine similarity.
4. Send retrieved context to a hosted instruction model (Groq, Google Gemini, or Hugging Face).
5. Display the answer and retrieved passages/page numbers.

## Files
- `app.py` — Streamlit interface with multi-provider AI support (Groq, Gemini, Hugging Face)
- `rag.py` — PDF parsing, chunking, retrieval, prompt construction
- `requirements.txt` — dependencies

## AI Providers (Why Groq is Recommended)
Hugging Face recently changed its Inference Providers router to require credits; free accounts quickly hit **`402 Payment Required`**.
To fix this, this app supports 100% free alternative providers:

| Provider | Cost | Speed | How to get Key |
|---|---|---|---|
| **⚡ Groq (Recommended)** | **100% Free** | **Ultra Fast (~500 T/s)** | [console.groq.com/keys](https://console.groq.com/keys) (No credit card needed) |
| **🤖 Google Gemini** | **Free (15 RPM)** | **Fast** | [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) |
| **🤗 Hugging Face** | Limited / Credits | Variable | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |

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

In the app sidebar, select **Groq** and paste your free Groq API key, or use Gemini / Hugging Face.

## Deploy to Streamlit Community Cloud
1. Push your code to GitHub.
2. Visit https://share.streamlit.io/ and sign in with GitHub.
3. Select **Create app**, choose repository `RAG_PDF_READER1`, branch `main`, and main file path `app.py`.
4. (Optional) In **Advanced Settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "gsk_..."
   # or
   GEMINI_API_KEY = "AIza..."
   ```
5. Deploy and start chatting with your PDF!
