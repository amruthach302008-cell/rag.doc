import streamlit as st
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import chromadb
import ollama

st.set_page_config(page_title="Mini RAG Q&A", page_icon="📚")

st.title("📚 Mini RAG Q&A")
st.write("Upload PDF documents and ask questions about them.")

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

client = chromadb.Client()
collection = client.get_or_create_collection(name="documents")

uploaded_files = st.file_uploader(
    "📄 Upload PDF file(s)",
    type=["pdf"],
    accept_multiple_files=True
)

if st.button("➕ Add Documents"):

    if not uploaded_files:
        st.warning("Please upload at least one PDF file.")

    else:
        total_chunks = 0

        for uploaded_file in uploaded_files:

            reader = PdfReader(uploaded_file)

            text = ""

            for page in reader.pages:
                page_text = page.extract_text()

                if page_text:
                    text += page_text + "\n"

            if not text.strip():
                st.warning(f"Could not extract text from {uploaded_file.name}.")
                continue

            chunks = [
                text[i:i + 500]
                for i in range(0, len(text), 500)
            ]

            embeddings = embedding_model.encode(chunks)

            ids = [
                f"{uploaded_file.name}_chunk_{i}"
                for i in range(len(chunks))
            ]

            collection.add(
                ids=ids,
                documents=chunks,
                embeddings=embeddings.tolist()
            )

            total_chunks += len(chunks)

        st.success(
            f"Added {total_chunks} chunk(s) from {len(uploaded_files)} PDF file(s) to ChromaDB."
        )


question = st.text_input(
    "❓ Ask a question about your PDF"
)

if st.button("🤖 Ask AI"):

    if not question.strip():
        st.warning("Please enter a question.")

    elif collection.count() == 0:
        st.warning("Please upload and add a PDF first.")

    else:

        question_embedding = embedding_model.encode([question])[0]

        results = collection.query(
            query_embeddings=[question_embedding.tolist()],
            n_results=min(3, collection.count())
        )

        retrieved_chunks = results["documents"][0]

        context = "\n\n".join(retrieved_chunks)

        prompt = f"""
You are a helpful AI assistant.

Answer the question ONLY using the context below.

Context:
{context}

Question:
{question}

If the answer is not present in the context,
say "I don't know based on the provided document."
"""

        try:

            response = ollama.chat(
                model="llama3.2",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            st.subheader("🤖 Answer")

            st.write(
                response["message"]["content"]
            )

            with st.expander("📖 Retrieved Context"):

                for i, chunk in enumerate(retrieved_chunks):

                    st.write(f"**Chunk {i + 1}:**")
                    st.write(chunk)

        except Exception as e:

            st.error(
                "Could not connect to Ollama. "
                "Make sure Ollama is running and "
                "run `ollama pull llama3.2` first."
            )

            st.code(str(e))


st.divider()

st.caption(
    "Python + Streamlit + PyPDF + Sentence Transformers + ChromaDB + Ollama"
)