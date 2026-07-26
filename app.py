"""ResearchMind AI — Entry point for the Streamlit application."""

import logging
from pathlib import Path

import streamlit as st

from backend.chat_memory import ChatMemory
from backend.chunker import TextChunker
from backend.citations import CitationBuilder
from backend.embeddings import EmbeddingClient, EmbeddingError
from backend.llm import LLMClient, LLMError
from backend.pdf_loader import PDFLoader, PDFLoadError
from backend.prompt_builder import PromptBuilder
from backend.retriever import Retriever, RetrievalError
from backend.vector_store import VectorStore

logging.basicConfig(level=logging.INFO)

# ── Constants ───────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

CUSTOM_CSS = """
<style>
    .stApp header {display: none;}
    .main-header {text-align: center; padding: 1rem 0;}
    .main-header h1 {font-size: 2.2rem; margin-bottom: 0.2rem;}
    .main-header p {color: #888; font-size: 1rem;}
    .chat-message {padding: 0.5rem 0;}
    .source-badge {
        display: inline-block;
        background: #f0f2f6;
        border-radius: 12px;
        padding: 2px 10px;
        font-size: 0.75rem;
        color: #555;
        margin-right: 4px;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
    }
    .stAlert {border-radius: 8px;}
</style>
"""

# ── Page Configuration ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="ResearchMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ── Cache expensive resources ──────────────────────────────────────────────

@st.cache_resource
def _load_embedding_model():
    """Load and warm up the SentenceTransformer model once."""
    client = EmbeddingClient()
    client.encode("warm-up")
    return client


# ── Initialization with status updates ────────────────────────────────────

init_status = st.status("Starting ResearchMind AI...", expanded=False)

if "llm_client" not in st.session_state:
    try:
        st.session_state.llm_client = LLMClient()
    except LLMError:
        st.session_state.llm_client = None

init_status.update(label="Loading embedding model...", state="running")
if "embedder" not in st.session_state:
    st.session_state.embedder = _load_embedding_model()
    init_status.update(label="Embedding model ready", state="complete")

init_status.update(label="Initializing services...", state="running")
if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore()

if "retriever" not in st.session_state:
    st.session_state.retriever = Retriever(
        st.session_state.embedder, st.session_state.vector_store
    )

if "prompt_builder" not in st.session_state:
    st.session_state.prompt_builder = PromptBuilder()

if "citation_builder" not in st.session_state:
    st.session_state.citation_builder = CitationBuilder()

if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = ChatMemory(max_exchanges=5)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs: dict[str, dict] = {}

init_status.update(label="Ready", state="complete")
init_status.empty()

# ── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    # ── API Key (cloud fallback) ─────────────────────────────────────
    if st.session_state.llm_client is None:
        st.markdown("## 🔑 Gemini API Key")
        api_key = st.text_input(
            "Enter your Gemini API key",
            type="password",
            placeholder="Paste your key here...",
            help="Get a free key at https://aistudio.google.com/app/apikey",
            label_visibility="collapsed",
        )
        if api_key:
            st.session_state.api_key = api_key
            try:
                st.session_state.llm_client = LLMClient(api_key=api_key)
                st.rerun()
            except LLMError as e:
                st.error(str(e))
        st.divider()

    st.markdown("## 📄 Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF research papers",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        loader = PDFLoader()
        chunker = TextChunker()
        progress_bar = st.progress(0, text="Preparing...")

        for i, uploaded_file in enumerate(uploaded_files):
            if uploaded_file.name in st.session_state.uploaded_docs:
                continue

            progress_bar.progress(
                (i + 0.5) / len(uploaded_files),
                text=f"Processing {uploaded_file.name}...",
            )

            try:
                safe_name = uploaded_file.name.replace(" ", "_")
                file_path = str(DATA_DIR / safe_name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                pages = loader.load(file_path)
                chunks = chunker.chunk_pages(pages)
                vectors = st.session_state.embedder.encode_batch(
                    [c.text for c in chunks]
                )
                st.session_state.vector_store.add(chunks, vectors)

                total_pages = pages[-1].total_pages if pages else 0
                total_chars = sum(len(p.text) for p in pages)

                st.session_state.uploaded_docs[uploaded_file.name] = {
                    "path": file_path,
                    "pages": total_pages,
                    "chunks": len(chunks),
                    "chars": total_chars,
                }

            except (PDFLoadError, FileNotFoundError, EmbeddingError) as e:
                st.error(f"**{uploaded_file.name}** — {e}")

            progress_bar.progress(
                (i + 1) / len(uploaded_files),
                text=f"Ready: {uploaded_file.name}",
            )

        progress_bar.empty()

    if st.session_state.uploaded_docs:
        st.divider()
        st.markdown("**Loaded Documents**")
        for doc_name, info in st.session_state.uploaded_docs.items():
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(
                    f"📄 **{doc_name}**  \n"
                    f"<span class='source-badge'>{info['pages']}p</span> "
                    f"<span class='source-badge'>{info['chunks']} chunks</span>",
                    unsafe_allow_html=True,
                )

        st.caption(f"Total: {st.session_state.vector_store.size} indexed chunks")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_memory.clear()
            st.rerun()
    with col2:
        if st.button("📂 Clear All", use_container_width=True, type="secondary"):
            st.session_state.messages = []
            st.session_state.chat_memory.clear()
            st.session_state.vector_store = VectorStore()
            st.session_state.uploaded_docs = {}
            st.rerun()

# ── Main Area ───────────────────────────────────────────────────────────────

st.markdown(
    '<div class="main-header">'
    "<h1>🧠 ResearchMind AI</h1>"
    "<p>Your AI-powered research assistant</p>"
    "</div>",
    unsafe_allow_html=True,
)

doc_count = len(st.session_state.uploaded_docs)
if doc_count > 0:
    st.markdown(
        f"<p style='text-align:center;color:#888;'>"
        f"{doc_count} document(s) loaded &mdash; ready for questions</p>",
        unsafe_allow_html=True,
    )
else:
    st.info("Upload PDF documents in the sidebar to start asking questions.", icon="📖")

# ── Chat History ────────────────────────────────────────────────────────────

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── Chat Input ──────────────────────────────────────────────────────────────

placeholder = (
    "Ask a question about your documents..."
    if st.session_state.vector_store.size > 0
    else "Upload PDFs first, then ask questions here..."
)

if prompt := st.chat_input(placeholder, disabled=doc_count == 0):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.chat_memory.add("user", prompt)

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if st.session_state.llm_client is None:
            st.error("LLM client is not available. Check your API key.")
        else:
            try:
                if st.session_state.vector_store.size > 0:
                    with st.status("🔍 Searching documents...", expanded=True) as status:
                        results = st.session_state.retriever.retrieve(prompt)
                        status.update(
                            label=f"✅ Found {len(results)} relevant passages",
                            state="complete",
                            expanded=False,
                        )

                    history = st.session_state.chat_memory.format_for_prompt()
                    rag_prompt = st.session_state.prompt_builder.build(
                        prompt, results, conversation_history=history
                    )

                    with st.expander("📚 Retrieved Context", expanded=False):
                        for chunk, score in results:
                            st.markdown(
                                f"<span class='source-badge'>{chunk.document_name} "
                                f"p.{chunk.page_number}</span> "
                                f"<span class='source-badge'>score: {score:.3f}</span>",
                                unsafe_allow_html=True,
                            )
                            st.text(chunk.text[:300] + ("..." if len(chunk.text) > 300 else ""))
                            st.divider()

                    full_response = st.write_stream(
                        st.session_state.llm_client.generate_stream(rag_prompt)
                    )

                    citations = st.session_state.citation_builder.build(results)
                    citation_md = st.session_state.citation_builder.to_markdown(citations)
                    st.markdown(citation_md)
                else:
                    full_response = st.write_stream(
                        st.session_state.llm_client.generate_stream(prompt)
                    )

                st.session_state.chat_memory.add("assistant", full_response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )
            except (LLMError, RetrievalError) as e:
                st.error(str(e))
