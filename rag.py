"""Simple PDF RAG backend for an AI lab.
Free local retrieval: TF-IDF. Generation: Hugging Face Inference API (free tier, rate limits apply).
"""
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

def read_pdf(file_obj):
    reader = PdfReader(file_obj)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"page": i, "text": text})
    if not pages:
        raise ValueError("No selectable text was found. This may be a scanned/image-only PDF; OCR is not included in this simple lab.")
    return pages

def chunk_pages(pages, chunk_words=180, overlap_words=40):
    chunks = []
    for page in pages:
        words = page["text"].split()
        step = max(1, chunk_words - overlap_words)
        for start in range(0, len(words), step):
            part = " ".join(words[start:start + chunk_words]).strip()
            if part:
                chunks.append({"text": part, "page": page["page"]})
    return chunks

class SimpleRAG:
    def __init__(self, chunks):
        if not chunks:
            raise ValueError("The PDF did not produce any text chunks.")
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in chunks])

    def retrieve(self, question, top_k=4):
        q = self.vectorizer.transform([question])
        scores = cosine_similarity(q, self.matrix).ravel()
        indices = scores.argsort()[::-1][:top_k]
        return [
            {**self.chunks[i], "score": float(scores[i])}
            for i in indices if scores[i] > 0
        ]

def make_prompt(question, passages):
    context = "\n\n".join(
        f"[Page {p['page']}]\n{p['text']}" for p in passages
    )
    return f"""You are a helpful teaching assistant for a Retrieval-Augmented Generation (RAG) lab.
Answer the question using ONLY the context below. If the context does not contain the answer,
say that the uploaded PDF does not provide enough information. Do not invent facts.
Cite page numbers in your answer, such as (Page 2).

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
