import streamlit as st
from huggingface_hub import InferenceClient
from rag import read_pdf, chunk_pages, SimpleRAG, make_prompt

st.set_page_config(page_title="PDF RAG Lab", page_icon="📚", layout="wide")
st.title("📚 PDF RAG Lab")
st.caption("Upload a PDF → chunk text → retrieve relevant passages → generate an answer.")
st.info("This demo uses free local TF-IDF retrieval and a Hugging Face hosted model. Free inference has rate limits; no hosted free model guarantees unlimited tokens.")

with st.sidebar:
    st.header("Model settings")
    hf_token = st.text_input("Hugging Face access token (free)", type="password",
        help="Create a free account and token at huggingface.co/settings/tokens. Do not put your token in GitHub.")
    model_id = st.text_input("Model ID", value="Qwen/Qwen2.5-7B-Instruct")
    top_k = st.slider("Retrieved chunks", 1, 8, 4)
    st.markdown("[Get a Hugging Face token](https://huggingface.co/settings/tokens)")

uploaded = st.file_uploader("Upload a text-based PDF", type=["pdf"])
if uploaded:
    try:
        # Cache the processed document for this Streamlit session.
        file_bytes = uploaded.getvalue()
        if st.session_state.get("file_name") != uploaded.name or st.session_state.get("file_bytes") != file_bytes:
            import io
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
            elif not hf_token:
                answer = "Relevant passages were retrieved, but no answer was generated because a Hugging Face token was not provided. Add your free token in the sidebar."
            else:
                try:
                    client = InferenceClient(model=model_id, token=hf_token)
                    response = client.chat_completion(
                        messages=[
                            {"role": "system", "content": "Answer only from the supplied PDF context. State when the context is insufficient. Cite page numbers."},
                            {"role": "user", "content": make_prompt(question, passages)}
                        ],
                        max_tokens=500,
                        temperature=0.2,
                    )
                    answer = response.choices[0].message.content
                except Exception as e:
                    answer = f"Model request failed: {e}\n\nThe retrieved PDF passages are still available below. Free model availability and limits can vary."
        st.session_state.chat.append({"q": question, "a": answer, "passages": passages})
    for item in reversed(st.session_state.get("chat", [])):
        with st.chat_message("user"):
            st.write(item["q"])
        with st.chat_message("assistant"):
            st.write(item["a"])
            if item["passages"]:
                with st.expander("View retrieved source passages"):
                    for p in item["passages"]:
                        st.markdown(f"**Page {p['page']} · similarity {p['score']:.3f}**")
                        st.write(p["text"])
                        st.divider()
else:
    st.markdown("**Lab workflow:** upload a PDF, then ask questions. The app extracts text, splits it into overlapping chunks, retrieves relevant chunks using TF-IDF, and sends only those passages to the language model.")
