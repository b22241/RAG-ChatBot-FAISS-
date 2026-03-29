import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

# Debug start
st.write("🚀 App is starting...")

# API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.warning("⚠️ GROQ_API_KEY missing (set in Render env)")
else:
    st.success("✅ API key loaded")

# Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="Chat with PDF", layout="wide")
st.title("💬 Chat with your PDF")

# Session
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

if uploaded_file and st.session_state.rag_chain is None:

    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Indexing..."):

        loader = PyPDFLoader("temp.pdf")
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(docs)

        embeddings = load_embeddings()

        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer only from context:\n{context}"),
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

    st.success("✅ PDF Indexed")

# Chat UI
for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    st.chat_message(role).write(msg.content)

query = st.chat_input("Ask...")

if query and st.session_state.rag_chain:

    st.chat_message("user").write(query)

    answer = st.session_state.rag_chain.invoke({
        "question": query,
        "chat_history": st.session_state.chat_history
    })

    st.chat_message("assistant").write(answer)

    st.session_state.chat_history.append(HumanMessage(content=query))
    st.session_state.chat_history.append(AIMessage(content=answer))