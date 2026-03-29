import streamlit as st
import os
import uuid
from dotenv import load_dotenv

# Loads from .env locally, Railway env vars in production
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import JinaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="Chat with PDF", layout="wide")
st.title("💬 Chat with your PDF")

# ✅ Give every browser session a unique ID
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

uploaded_file = st.file_uploader("Upload PDF", type="pdf")


@st.cache_resource
def load_embeddings():
    jina_api_key = os.getenv("JINA_API_KEY")
    if not jina_api_key:
        st.error("❌ JINA_API_KEY missing — add it to your .env file or Railway env vars")
        return None
    return JinaEmbeddings(
        jina_api_key=jina_api_key,
        model_name="jina-embeddings-v2-base-en"
    )


if uploaded_file and st.session_state.rag_chain is None:

    # ✅ Each user writes to their own unique temp file
    temp_path = f"temp_{st.session_state.session_id}.pdf"

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Indexing your PDF..."):

        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(docs)

        embeddings = load_embeddings()

        if embeddings is None:
            st.stop()

        # ✅ Vectorstore is stored inside session_state — isolated per user
        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer only from the provided context:\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}")
        ])

        def format_docs(docs):
            return "\n\n".join(d.page_content for d in docs)

        def get_context(inputs):
            return format_docs(retriever.invoke(inputs["question"]))

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

    # ✅ Clean up temp file after indexing to save disk space
    if os.path.exists(temp_path):
        os.remove(temp_path)

    st.success("✅ PDF indexed! Start chatting below.")

# Chat UI
for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    st.chat_message(role).write(msg.content)

query = st.chat_input("Ask something about your PDF...")

if query and st.session_state.rag_chain:

    st.chat_message("user").write(query)

    answer = st.session_state.rag_chain.invoke({
        "question": query,
        "chat_history": st.session_state.chat_history
    })

    st.chat_message("assistant").write(answer)

    st.session_state.chat_history.append(HumanMessage(content=query))
    st.session_state.chat_history.append(AIMessage(content=answer))

elif query and not st.session_state.rag_chain:
    st.warning("⚠️ Please upload a PDF first.")