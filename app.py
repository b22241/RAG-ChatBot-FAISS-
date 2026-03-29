import streamlit as st
import os
from dotenv import load_dotenv

# Load env variables (works locally, ignored on Render)
load_dotenv()

# Get PORT (Render provides it, default for local)
PORT = int(os.environ.get("PORT", 8501))

# Check API Key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found. Set it in environment variables.")
    st.stop()

# LangChain imports
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# ─── Streamlit Config ─────────────────────────────────────────
st.set_page_config(page_title="Chat with PDF", layout="wide")
st.title("💬 Chat with your PDF (RAG + Groq)")

# ─── Session State ─────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# ─── Upload PDF ─────────────────────────────────────────
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file and st.session_state.rag_chain is None:

    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Indexing your PDF..."):

        # Load PDF
        loader = PyPDFLoader("temp.pdf")
        documents = loader.load()

        # Split
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(documents)

        # Embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Vector Store
        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 4}
        )

        # LLM
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0
        )

        # Prompt
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a precise research assistant.

Rules:
- Answer only from context
- Use bullet points
- If not found, say: "I couldn't find that in the document."

Context:
{context}"""
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}")
        ])

        # Helper functions
        def format_docs(docs):
            return "\n\n".join(d.page_content for d in docs)

        def get_context(inputs):
            docs = retriever.invoke(inputs["question"])
            st.session_state.last_docs = docs
            return format_docs(docs)

        # RAG Chain
        rag_chain = (
            {
                "context": get_context,
                "question": lambda x: x["question"],
                "chat_history": lambda x: x["chat_history"],
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        st.session_state.rag_chain = rag_chain

    st.success(f"✅ Indexed {len(chunks)} chunks")

# ─── Chat UI ─────────────────────────────────────────
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    else:
        st.chat_message("assistant").write(msg.content)

query = st.chat_input("Ask something...")

if query and st.session_state.rag_chain:

    st.chat_message("user").write(query)

    with st.spinner("Thinking..."):
        answer = st.session_state.rag_chain.invoke({
            "question": query,
            "chat_history": st.session_state.chat_history
        })

    st.chat_message("assistant").write(answer)

    st.session_state.chat_history.append(HumanMessage(content=query))
    st.session_state.chat_history.append(AIMessage(content=answer))

elif query:
    st.warning("⚠️ Upload a PDF first")