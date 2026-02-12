import os
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Healthcare AI Assistant", page_icon="🩺")
st.title("Healthcare AI Assistant 🩺")
st.write("Upload a medical PDF and ask questions from it (RAG).")
st.warning("This tool summarizes documents and is not medical advice. For clinical decisions, consult a qualified professional.")

api_key = st.text_input("Enter your OpenAI API key:", type="password")
uploaded_file = st.file_uploader("Upload a medical PDF", type=["pdf"])

# -----------------------------
# Helper functions
# -----------------------------
def page_display_from_doc(doc) -> str:
    """Convert 0-indexed page to human-friendly 1-indexed page."""
    page = doc.metadata.get("page", None)
    if isinstance(page, int):
        return str(page + 1)
    return "N/A"

def format_context_with_page_tags(retrieved_docs) -> str:
    """Create context that keeps page tags so the model can cite (p. X)."""
    blocks = []
    for d in retrieved_docs:
        p = page_display_from_doc(d)
        blocks.append(f"[p. {p}]\n{d.page_content}")
    return "\n\n---\n\n".join(blocks)

# -----------------------------
# Main logic
# -----------------------------
if api_key and uploaded_file:
    # Set key for this session
    os.environ["OPENAI_API_KEY"] = api_key

    # Save uploaded PDF locally
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Load PDF
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    docs = splitter.split_documents(documents)

    # Build vector store (embeddings)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 4})

    # LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    query = st.text_input("Ask a question from the PDF:")

    if query:
        with st.spinner("Searching and thinking..."):
            # New LangChain style: use invoke()
            retrieved_docs = retriever.invoke(query)

            # Show retrieved chunks (verification)
            with st.expander("Show retrieved PDF chunks"):
                for i, d in enumerate(retrieved_docs, 1):
                    p = page_display_from_doc(d)
                    st.markdown(f"**Chunk {i} (page {p})**")
                    st.write(d.page_content[:1200])
                    st.write("---")

            # Build context with page tags for citations
            context = format_context_with_page_tags(retrieved_docs)

            prompt = f"""
You are a careful assistant answering questions from a PDF.
Use ONLY the context below. Do NOT add outside knowledge.
If the answer is not in the context, say: "Not found in the document."

Rules:
- Answer in bullet points when possible.
- After each bullet point, add a short citation like (p. X).
- The context already includes page tags like [p. 3]. Use those pages.

CONTEXT:
{context}

QUESTION:
{query}
"""

            answer = llm.invoke(prompt).content

        st.subheader("Answer")
        st.write(answer)

else:
    st.info("Step 1: Enter OpenAI API key. Step 2: Upload PDF.")

