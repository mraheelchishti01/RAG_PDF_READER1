import io
import requests
import streamlit as st
from rag import read_pdf, chunk_pages, SimpleRAG, make_prompt

st.set_page_config(page_title="PDF RAG Lab", page_icon="📚", layout="wide")
st.title("📚 PDF RAG Lab")
st.caption("Upload a PDF → chunk text → retrieve relevant passages → generate an answer.")
st.info("Local TF-IDF retrieval + Free AI Generation. Use Groq or Gemini for reliable 100% free access without credit limits.")


def call_groq(api_key: str, model: str, system_prompt: str, user_prompt: str) -> str:
    """Calls Groq OpenAI-compatible chat completions endpoint (100% Free tier)."""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 600,
    }
    res = requests.post(url, headers=headers, json=payload, timeout=30)
    if res.status_code != 200:
        try:
            err_msg = res.json().get("error", {}).get("message", res.text)
        except Exception:
            err_msg = res.text
        raise RuntimeError(f"Groq API Error ({res.status_code}): {err_msg}")
    return res.json()["choices"][0]["message"]["content"]


def call_gemini(api_key: str, model: str, system_prompt: str, user_prompt: str) -> str:
    """Calls Google Gemini generateContent REST endpoint (Free tier)."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key.strip()}"
    headers = {"Content-Type": "application/json"}
    full_prompt = f"System Instructions:\n{system_prompt}\n\nTask and Context:\n{user_prompt}"
    payload = {
        "contents": [{"parts": [{"text": full_prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 600},
    }
    res = requests.post(url, headers=headers, json=payload, timeout=30)
    if res.status_code != 200:
        try:
            err_msg = res.json().get("error", {}).get("message", res.text)
        except Exception:
            err_msg = res.text
        raise RuntimeError(f"Gemini API Error ({res.status_code}): {err_msg}")
    data = res.json()
    candidates = data.get("candidates", [])
    if candidates and "content" in candidates[0]:
        parts = candidates[0]["content"].get("parts", [])
        if parts and "text" in parts[0]:
            return parts[0]["text"]
    raise RuntimeError("Gemini did not return any text response.")


def call_huggingface(api_key: str, model: str, system_prompt: str, user_prompt: str) -> str:
    """Calls Hugging Face InferenceClient chat_completion."""
    from huggingface_hub import InferenceClient

    client = InferenceClient(model=model, token=api_key.strip())
    response = client.chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=600,
        temperature=0.2,
    )
    return response.choices[0].message.content


with st.sidebar:
    st.header("⚙️ Model Settings")
    provider = st.selectbox(
        "Choose AI Provider",
        [
            "Groq (100% Free & Fast ⚡)",
            "Google Gemini (Free 🤖)",
            "Hugging Face 🤗",
        ],
        index=0,
        help="Groq and Gemini have generous free tiers with no credit cards required. Hugging Face free tier runs out quickly.",
    )

    if "Groq" in provider:
        default_groq = st.secrets.get("GROQ_API_KEY", "") if hasattr(st, "secrets") else ""
        api_key = st.text_input(
            "Groq API Key (Free)",
            value=default_groq,
            type="password",
            help="Free key at https://console.groq.com/keys (No credit card needed)",
        )
        model_id = st.selectbox(
            "Groq Model",
            [
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
                "mixtral-8x7b-32768",
                "gemma2-9b-it",
            ],
            index=0,
        )
        st.markdown("[👉 Get Free Groq API Key](https://console.groq.com/keys)")
        st.caption("✅ Fast, accurate, and completely free.")

    elif "Gemini" in provider:
        default_gemini = st.secrets.get("GEMINI_API_KEY", "") if hasattr(st, "secrets") else ""
        api_key = st.text_input(
            "Gemini API Key (Free)",
            value=default_gemini,
            type="password",
            help="Free key at https://aistudio.google.com/app/apikey",
        )
        model_id = st.selectbox(
            "Gemini Model",
            ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"],
            index=0,
        )
        st.markdown("[👉 Get Free Gemini API Key](https://aistudio.google.com/app/apikey)")
        st.caption("✅ Google free tier (15 requests/min).")

    else:
        default_hf = st.secrets.get("HF_TOKEN", "") if hasattr(st, "secrets") else ""
        api_key = st.text_input(
            "Hugging Face Token",
            value=default_hf,
            type="password",
            help="Create a token at https://huggingface.co/settings/tokens",
        )
        model_id = st.text_input(
            "Hugging Face Model ID",
            value="Qwen/Qwen2.5-7B-Instruct",
            help="External router models require credits. If you see 402, use Groq instead.",
        )
        st.markdown("[👉 Get Hugging Face Token](https://huggingface.co/settings/tokens)")
        st.warning("⚠️ Hugging Face free tier gives 402 Payment Required once router credits ($0.10) expire.")

    top_k = st.slider("Retrieved chunks", 1, 8, 4)

uploaded = st.file_uploader("Upload a text-based PDF", type=["pdf"])
if uploaded:
    try:
        file_bytes = uploaded.getvalue()
        if st.session_state.get("file_name") != uploaded.name or st.session_state.get("file_bytes") != file_bytes:
            pages = read_pdf(io.BytesIO(file_bytes))
            chunks = chunk_pages(pages)
            st.session_state.rag = SimpleRAG(chunks)
            st.session_state.file_name = uploaded.name
            st.session_state.file_bytes = file_bytes
            st.session_state.page_count = len(pages)
            st.session_state.chunk_count = len(chunks)
            st.session_state.chat = []
        st.success(f"PDF ready: {uploaded.name} · {st.session_state.page_count} text pages · {st.session_state.chunk_count} chunks")
    except Exception as e:
        st.error(f"Could not process PDF: {e}")

if "rag" in st.session_state:
    st.subheader("Ask a question about your PDF")
    question = st.chat_input("Type a question grounded in the uploaded PDF...")
    if question:
        with st.spinner("Retrieving passages and generating an answer..."):
            passages = st.session_state.rag.retrieve(question, top_k=top_k)
            if not passages:
                answer = "I could not find a relevant passage in the uploaded PDF. Try rephrasing the question."
            elif not api_key:
                provider_name = provider.split()[0]
                answer = (
                    f"⚠️ Relevant passages were retrieved, but no API key was provided for **{provider_name}**.\n\n"
                    f"Please enter your key in the sidebar."
                )
            else:
                system_prompt = (
                    "Answer only from the supplied PDF context. State when the context is insufficient. Cite page numbers."
                )
                user_prompt = make_prompt(question, passages)
                try:
                    if "Groq" in provider:
                        answer = call_groq(api_key, model_id, system_prompt, user_prompt)
                    elif "Gemini" in provider:
                        answer = call_gemini(api_key, model_id, system_prompt, user_prompt)
                    else:
                        answer = call_huggingface(api_key, model_id, system_prompt, user_prompt)
                except Exception as e:
                    err_msg = str(e)
                    if "402" in err_msg or "Payment Required" in err_msg:
                        answer = (
                            "❌ **402 Payment Required: Hugging Face Credits Exhausted**\n\n"
                            "Aapke Hugging Face account ke free inference credits ($0.10) khatam ho chuke hain, is liye router request reject kar raha hai.\n\n"
                            "💡 **Aasan Hal (Fix):**\n"
                            "1. Sidebar mein **Choose AI Provider** dropdown se **`Groq (100% Free & Fast ⚡)`** select karein.\n"
                            "2. [console.groq.com/keys](https://console.groq.com/keys) se 30 second mein free API key le kar sidebar mein paste karein (koi credit card nahi chahiye).\n"
                            "3. Groq par `llama-3.3-70b-versatile` super-fast aur unlimited free chalega!\n\n"
                            f"Technical Error Details: `{err_msg}`"
                        )
                    else:
                        answer = f"Model request failed: {e}\n\nThe retrieved PDF passages are still available below."
        st.session_state.chat.append({"q": question, "a": answer, "passages": passages})

    for item in reversed(st.session_state.get("chat", [])):
        with st.chat_message("user"):
            st.write(item["q"])
        with st.chat_message("assistant"):
            st.write(item["a"])
            if item.get("passages"):
                with st.expander("View retrieved source passages"):
                    for p in item["passages"]:
                        st.markdown(f"**Page {p['page']} · similarity {p['score']:.3f}**")
                        st.write(p["text"])
                        st.divider()
else:
    st.markdown("**Lab workflow:** upload a PDF, then ask questions. The app extracts text, splits it into overlapping chunks, retrieves relevant chunks using TF-IDF, and sends only those passages to the language model.")
