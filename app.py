import streamlit as st
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage

# ─── Config ───────────────────────────────────────────────────────────────────
os.environ["GROQ_API_KEY"] = "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"

st.set_page_config(page_title="Chat with PDF", layout="wide")
st.title("💬 Chat with your PDF (LCEL + Custom Prompt)")

# ─── Session State ─────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []        # List of HumanMessage / AIMessage

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

# ─── PDF Upload & Indexing ─────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file and st.session_state.rag_chain is None:

    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Indexing your PDF..."):

        # 1. Load
        loader = PyPDFLoader("temp.pdf")
        documents = loader.load()

        # 2. Split
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(documents)

        # 3. Embed
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # 4. Vector store  ── MMR for diverse retrieval
        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.6}
        )
        st.session_state.retriever = retriever

        # 5. LLM
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

        # ── CHANGE 2: Your own ChatPromptTemplate ─────────────────────────────
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a precise research assistant.
Your job is to answer questions strictly based on the document context provided below.

Rules:
- Answer in clear, concise bullet points when listing facts.
- If the answer is not present in the context, respond with:
  "I couldn't find that in the document."
- Never make up information. Never use outside knowledge.
- Keep answers focused and under 150 words unless the user asks for detail.

Context from the document:
{context}"""
            ),
            MessagesPlaceholder("chat_history"),   # ← memory injected here
            ("human", "{question}"),
        ])

        # ── CHANGE 1: LCEL chain with | operator ──────────────────────────────
        def format_docs(docs):
            return "\n\n".join(
                f"[Page {d.metadata.get('page', '?')}] {d.page_content}"
                for d in docs
            )

        def get_context(inputs):
            # Retrieves docs using the question, returns formatted string
            docs = retriever.invoke(inputs["question"])
            # Store docs in session for source display
            st.session_state.last_source_docs = docs
            return format_docs(docs)

        rag_chain = (
            {
                "context":      get_context,            # retrieve + format
                "question":     lambda x: x["question"],
                "chat_history": lambda x: x["chat_history"],
            }
            | prompt            # fill the template
            | llm               # call Groq
            | StrOutputParser() # pull plain string out of AIMessage
        )

        st.session_state.rag_chain = rag_chain

    st.success(f"✅ Indexed {len(chunks)} chunks from your PDF!")

# ─── Display Chat History ──────────────────────────────────────────────────────
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)

# ─── Chat Input ───────────────────────────────────────────────────────────────
query = st.chat_input("Ask something from the PDF...")

if query and st.session_state.rag_chain:

    # Show user message immediately
    st.chat_message("user").write(query)

    with st.spinner("Thinking..."):
        # ── CHANGE 3: Pass HumanMessage/AIMessage history ─────────────────────
        answer = st.session_state.rag_chain.invoke({
            "question": query,
            "chat_history": st.session_state.chat_history  # list of message objects
        })

    # Display answer
    st.chat_message("assistant").write(answer)

    # ── CHANGE 3: Store as message objects, not (q, a) tuples ─────────────────
    st.session_state.chat_history.append(HumanMessage(content=query))
    st.session_state.chat_history.append(AIMessage(content=answer))

    # ── Source documents ───────────────────────────────────────────────────────
    if hasattr(st.session_state, "last_source_docs"):
        st.write("### 📄 Sources")
        for i, doc in enumerate(st.session_state.last_source_docs):
            page = doc.metadata.get("page", "?")
            with st.expander(f"Source {i+1} — Page {page}"):
                st.write(doc.page_content)

elif query and not st.session_state.rag_chain:
    st.warning("⚠️ Please upload a PDF first.")