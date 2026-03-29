import streamlit as st
import os
import uuid
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import JinaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(
    page_title="DocMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background: #040810 !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 50% at 50% -20%, rgba(0,200,255,0.08) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 100%, rgba(100,0,255,0.06) 0%, transparent 60%),
        #040810 !important;
}

/* Scrolling star dots background */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        radial-gradient(1px 1px at 10% 15%, rgba(0,200,255,0.3) 0%, transparent 100%),
        radial-gradient(1px 1px at 30% 70%, rgba(150,100,255,0.3) 0%, transparent 100%),
        radial-gradient(1px 1px at 60% 25%, rgba(0,200,255,0.2) 0%, transparent 100%),
        radial-gradient(1px 1px at 85% 60%, rgba(100,200,255,0.2) 0%, transparent 100%),
        radial-gradient(1px 1px at 45% 90%, rgba(200,100,255,0.2) 0%, transparent 100%),
        radial-gradient(1.5px 1.5px at 75% 10%, rgba(0,255,200,0.3) 0%, transparent 100%),
        radial-gradient(1px 1px at 20% 45%, rgba(0,200,255,0.15) 0%, transparent 100%),
        radial-gradient(1px 1px at 92% 35%, rgba(150,100,255,0.2) 0%, transparent 100%);
    pointer-events: none;
    z-index: 0;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }

/* ── Main container ── */
.main .block-container {
    max-width: 860px !important;
    padding: 2rem 2rem 6rem !important;
    margin: 0 auto !important;
    position: relative;
    z-index: 1;
}

/* ── Hero header ── */
.hero {
    text-align: center;
    padding: 3rem 0 2.5rem;
    position: relative;
}

.hero-badge {
    display: inline-block;
    background: rgba(0,200,255,0.08);
    border: 1px solid rgba(0,200,255,0.25);
    color: #00c8ff;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    padding: 0.3rem 1rem;
    border-radius: 100px;
    margin-bottom: 1.2rem;
    animation: fadeSlideDown 0.6s ease both;
}

.hero-title {
    font-size: clamp(2.2rem, 5vw, 3.5rem);
    font-weight: 700;
    line-height: 1.1;
    margin-bottom: 0.8rem;
    background: linear-gradient(135deg, #ffffff 0%, #00c8ff 50%, #a855f7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: fadeSlideDown 0.6s ease 0.1s both;
}

.hero-sub {
    color: rgba(255,255,255,0.4);
    font-size: 1rem;
    font-weight: 400;
    animation: fadeSlideDown 0.6s ease 0.2s both;
}

/* ── Upload zone ── */
.upload-wrapper {
    background: rgba(255,255,255,0.02);
    border: 1.5px dashed rgba(0,200,255,0.25);
    border-radius: 16px;
    padding: 2rem;
    margin: 1.5rem 0;
    transition: border-color 0.3s, background 0.3s;
    animation: fadeSlideDown 0.6s ease 0.3s both;
    position: relative;
    overflow: hidden;
}

.upload-wrapper::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(0,200,255,0.03) 0%, rgba(168,85,247,0.03) 100%);
    opacity: 0;
    transition: opacity 0.3s;
}

.upload-wrapper:hover { border-color: rgba(0,200,255,0.5); }
.upload-wrapper:hover::before { opacity: 1; }

[data-testid="stFileUploader"] {
    background: transparent !important;
    border: none !important;
}

[data-testid="stFileUploader"] label {
    color: rgba(255,255,255,0.6) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.9rem !important;
}

[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

[data-testid="stFileUploaderDropzoneInstructions"] {
    color: rgba(255,255,255,0.5) !important;
}

/* Upload button */
[data-testid="stFileUploaderDropzone"] button {
    background: linear-gradient(135deg, #00c8ff22, #a855f722) !important;
    border: 1px solid rgba(0,200,255,0.4) !important;
    color: #00c8ff !important;
    border-radius: 8px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 500 !important;
    transition: all 0.2s !important;
}

[data-testid="stFileUploaderDropzone"] button:hover {
    background: linear-gradient(135deg, #00c8ff33, #a855f733) !important;
    border-color: rgba(0,200,255,0.7) !important;
    box-shadow: 0 0 20px rgba(0,200,255,0.2) !important;
}

/* ── Status messages ── */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    border: none !important;
}

.stSuccess {
    background: rgba(0,255,150,0.08) !important;
    border: 1px solid rgba(0,255,150,0.2) !important;
    color: #00ff96 !important;
}

.stWarning {
    background: rgba(255,170,0,0.08) !important;
    border: 1px solid rgba(255,170,0,0.2) !important;
}

.stError {
    background: rgba(255,50,50,0.08) !important;
    border: 1px solid rgba(255,50,50,0.2) !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] {
    color: #00c8ff !important;
}

/* ── Divider ── */
.chat-divider {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin: 1.5rem 0;
    animation: fadeSlideDown 0.6s ease 0.4s both;
}
.chat-divider::before, .chat-divider::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,200,255,0.2), transparent);
}
.chat-divider span {
    color: rgba(255,255,255,0.25);
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    white-space: nowrap;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.5rem 0 !important;
    animation: msgIn 0.4s cubic-bezier(0.16,1,0.3,1) both;
}

@keyframes msgIn {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* User message bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: linear-gradient(135deg, rgba(0,200,255,0.06), rgba(0,150,200,0.04)) !important;
    border: 1px solid rgba(0,200,255,0.12) !important;
    border-radius: 16px 16px 4px 16px !important;
    padding: 1rem 1.2rem !important;
    margin-left: 3rem !important;
}

/* Assistant message bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: linear-gradient(135deg, rgba(168,85,247,0.06), rgba(120,60,200,0.04)) !important;
    border: 1px solid rgba(168,85,247,0.12) !important;
    border-radius: 16px 16px 16px 4px !important;
    padding: 1rem 1.2rem !important;
    margin-right: 3rem !important;
}

/* Avatar icons */
[data-testid="chatAvatarIcon-user"] {
    background: linear-gradient(135deg, #00c8ff, #0080ff) !important;
    border-radius: 10px !important;
}

[data-testid="chatAvatarIcon-assistant"] {
    background: linear-gradient(135deg, #a855f7, #6020c0) !important;
    border-radius: 10px !important;
}

/* Message text */
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span {
    color: rgba(255,255,255,0.85) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.95rem !important;
    line-height: 1.7 !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    position: fixed !important;
    bottom: 0 !important;
    left: 50% !important;
    transform: translateX(-50%) !important;
    width: min(860px, 100vw) !important;
    padding: 1rem 2rem 1.5rem !important;
    background: linear-gradient(to top, #040810 70%, transparent) !important;
    z-index: 999 !important;
}

[data-testid="stChatInput"] > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1.5px solid rgba(0,200,255,0.2) !important;
    border-radius: 14px !important;
    backdrop-filter: blur(20px) !important;
    transition: border-color 0.3s, box-shadow 0.3s !important;
}

[data-testid="stChatInput"] > div:focus-within {
    border-color: rgba(0,200,255,0.5) !important;
    box-shadow: 0 0 30px rgba(0,200,255,0.1), 0 0 0 1px rgba(0,200,255,0.1) !important;
}

[data-testid="stChatInput"] textarea {
    color: rgba(255,255,255,0.9) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.95rem !important;
    background: transparent !important;
    caret-color: #00c8ff !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: rgba(255,255,255,0.25) !important;
}

[data-testid="stChatInputSubmitButton"] {
    background: linear-gradient(135deg, #00c8ff, #a855f7) !important;
    border-radius: 8px !important;
    border: none !important;
    transition: opacity 0.2s, transform 0.2s !important;
}

[data-testid="stChatInputSubmitButton"]:hover {
    opacity: 0.85 !important;
    transform: scale(0.96) !important;
}

/* ── Spinner text ── */
.stSpinner p {
    color: rgba(255,255,255,0.5) !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

/* ── Animations ── */
@keyframes fadeSlideDown {
    from { opacity: 0; transform: translateY(-16px); }
    to   { opacity: 1; transform: translateY(0); }
}

@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 20px rgba(0,200,255,0.15); }
    50%       { box-shadow: 0 0 40px rgba(0,200,255,0.3); }
}

/* ── Stats bar ── */
.stats-bar {
    display: flex;
    gap: 1rem;
    justify-content: center;
    margin: 1rem 0 1.5rem;
    animation: fadeSlideDown 0.6s ease 0.5s both;
}

.stat-pill {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 100px;
    padding: 0.3rem 0.9rem;
    font-size: 0.72rem;
    color: rgba(255,255,255,0.35);
    letter-spacing: 0.05em;
}

.stat-pill span {
    color: #00c8ff;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# ── Hero ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">🧠 Powered by LLaMA 3.3 · Groq · Jina AI</div>
    <div class="hero-title">DocMind AI</div>
    <div class="hero-sub">Upload any PDF and have an intelligent conversation with it</div>
</div>
""", unsafe_allow_html=True)

# ── Stats bar ──────────────────────────────────────────────────────────────
msg_count = len(st.session_state.chat_history) // 2
pdf_status = st.session_state.pdf_name or "No PDF"
st.markdown(f"""
<div class="stats-bar">
    <div class="stat-pill">📄 <span>{pdf_status}</span></div>
    <div class="stat-pill">💬 <span>{msg_count}</span> messages</div>
    <div class="stat-pill">⚡ <span>LLaMA 3.3 70B</span></div>
</div>
""", unsafe_allow_html=True)

# ── Embeddings ─────────────────────────────────────────────────────────────
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

# ── Upload zone ────────────────────────────────────────────────────────────
st.markdown('<div class="upload-wrapper">', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Drop your PDF here",
    type="pdf",
    label_visibility="collapsed"
)
st.markdown('</div>', unsafe_allow_html=True)

# ── PDF Indexing ───────────────────────────────────────────────────────────
if uploaded_file and st.session_state.rag_chain is None:

    temp_path = f"temp_{st.session_state.session_id}.pdf"

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    st.session_state.pdf_name = uploaded_file.name

    with st.spinner(f"🔍 Indexing **{uploaded_file.name}**..."):

        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)

        embeddings = load_embeddings()
        if embeddings is None:
            st.stop()

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

    if os.path.exists(temp_path):
        os.remove(temp_path)

    st.success(f"✅ **{uploaded_file.name}** indexed — {len(chunks)} chunks ready. Start chatting!")
    st.rerun()

# ── Chat UI ────────────────────────────────────────────────────────────────
if st.session_state.chat_history:
    st.markdown("""
    <div class="chat-divider"><span>conversation</span></div>
    """, unsafe_allow_html=True)

for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    st.chat_message(role).write(msg.content)

# ── Input ──────────────────────────────────────────────────────────────────
placeholder = "Ask anything about your PDF..." if st.session_state.rag_chain else "Upload a PDF above to start chatting..."
query = st.chat_input(placeholder)

if query and st.session_state.rag_chain:

    st.chat_message("user").write(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = st.session_state.rag_chain.invoke({
                "question": query,
                "chat_history": st.session_state.chat_history
            })
        st.write(answer)

    st.session_state.chat_history.append(HumanMessage(content=query))
    st.session_state.chat_history.append(AIMessage(content=answer))
    st.rerun()

elif query and not st.session_state.rag_chain:
    st.warning("⚠️ Please upload a PDF first.")