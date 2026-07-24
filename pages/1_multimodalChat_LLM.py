
import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import streamlit as st
import requests
import base64
from langchain_core.messages import HumanMessage, SystemMessage

from src.phaseOne.chunking import create_chunking
from src.phaseTwo.prompt_llm import create_system_prompt, prompt_is_ok, stream_ollama_response
from src.phaseTwo.retrieval import retrival_to_LLM

load_dotenv()
HOST_OLLAMA=os.getenv('HOST_OLLAMA')
PORT_OLLAMA=os.getenv('PORT_OLLAMA')

# ── Configurazione pagina ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chat LLM · AI Studio",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS globale e Tema Moderno ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Outfit:wght@700&display=swap');

:root {
  --text-main: #111827;
  --text-muted: #6B7280;
  --bg-light: #F9FAFB;
  --card-bg: #FFFFFF;
  --border-light: #E5E7EB;
  --accent-start: #4F46E5;
  --accent-end: #7C3AED;
  --accent-gradient: linear-gradient(135deg, var(--accent-start) 0%, var(--accent-end) 100%);
}

html, body, [class*="css"] {
  font-family: 'Inter', sans-serif !important;
}

/* Stili Header */
.main-title {
  font-family: 'Outfit', sans-serif;
  font-size: 3.5rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: var(--text-main);
  line-height: 1.1;
  margin-top: -1rem;
}

.main-title span {
  background: var(--accent-gradient);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.subtitle {
  color: var(--text-muted);
  font-size: 1rem;
  font-weight: 500;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  margin-bottom: 2rem;
  margin-top: 0.5rem;
}

/* Stile per i messaggi utente/assistente */
div[data-testid="stChatMessage"] {
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 0.5rem;
}

/* Stile speciale per l'uploader delle immagini */
div[data-testid="stFileUploader"] {
    border: 2px dashed #D1D5DB !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    background-color: #F9FAFB !important;
    transition: all 0.2s ease !important;
}

div[data-testid="stFileUploader"]:hover {
    border-color: #8B5CF6 !important;
    background-color: #F5F3FF !important;
}

/* Bottoni Sidebar */
div[data-testid="stSidebar"] div[data-testid="stButton"] button {
    border-radius: 8px !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Helper: recupera modelli da Ollama ───────────────────────────────────────
@st.cache_data(ttl=30)
def get_ollama_models():
    try:
        r = requests.get(f"http://{HOST_OLLAMA}:{PORT_OLLAMA}/api/tags", timeout=3)
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


# ── Inizializzazione stato ────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "images" not in st.session_state:
    st.session_state.images = None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <h1 style="font-family:'Outfit', sans-serif; font-size:2.2rem; font-weight:700; color:#111827; margin-bottom:0;">
            AI <span style="background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">STUDIO</span>
        </h1>
        <div style="color:#6B7280; font-size:0.85rem; font-weight:500; margin-bottom:2rem; letter-spacing:0.05em;">PANNELLO DI CONTROLLO</div>
    """, unsafe_allow_html=True)

    available_models = get_ollama_models()
    ollama_ok = len(available_models) > 0

    if ollama_ok:
        model_sel = st.selectbox("Modello Attivo", options=available_models, index=0)
        st.success("🟢 Ollama Online")
    else:
        model_sel = "llama3"
        st.error("🔴 Ollama Offline. Avvia il server con `ollama serve`.")

    st.divider()
    temperature = st.slider("Temperatura (Creatività)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

    st.divider()
    if st.button("🗑️ Svuota Memoria Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.images = None
        st.rerun()

# ── Header pagina ─────────────────────────────────────────────────────────────
st.markdown("<div class='main-title'>CHAT <span>MULTIMODALE</span></div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Conversazione locale · Nessun dato in cloud</div>", unsafe_allow_html=True)

# ── Area messaggi (Nativa di Streamlit) ───────────────────────────────────────
for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        # Se c'è un'immagine salvata nello stato, mostrala
        if msg.get("image"):
            st.image(base64.b64decode(msg["image"]), width=300)

# ── Area Input Multimodale (Pagina Principale) ────────────────────────────────
# Usiamo un expander per mantenere l'interfaccia pulita, l'utente lo apre solo se serve caricare un'immagine
if st.session_state.images is None:
    with st.expander("📎 Allega un'immagine per la prossima domanda", expanded=True):
        uploaded_image = st.file_uploader("Trascina qui l'immagine o sfoglia i file", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
        st.session_state.images = uploaded_image

# ── Input Utente & Generazione ────────────────────────────────────────────────
prompt_to_send = None

# Input principale via chat nativa
if chat_input := st.chat_input("Scrivi un messaggio...", disabled=not ollama_ok):
    prompt_to_send = chat_input

# Logica di elaborazione

if prompt_is_ok(prompt_to_send) and st.session_state.images:

    img_base64 = base64.b64encode(st.session_state.images.read()).decode("utf-8")

    if len(st.session_state.messages) == 0:
        image_query_bytes = st.session_state.images.getvalue()
        prompt_humanMessage = retrival_to_LLM(prompt_to_send,image_query_bytes)
        st.session_state.messages.append({"role": "user", "content": prompt_to_send, "image": img_base64})
    else:
        prompt_humanMessage = HumanMessage(content=prompt_to_send)
        st.session_state.messages.append({"role": "user", "content": prompt_to_send, "image": None})

    system_prompt = create_system_prompt()

    if not ollama_ok:
        st.error("⚠️ Ollama non è raggiungibile.")
        st.stop()

    # Mostra messaggio utente istantaneamente nell'interfaccia


    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt_to_send)
        if img_base64:
            st.image(st.session_state.images, width=300)

    # Genera la risposta con streaming
    with st.chat_message("assistant", avatar="🤖"):
        stream_placeholder = st.empty()
        full_reply = ""

        # History non include l'ultimo prompt_to_send (appena aggiunto)
        history_for_llm = st.session_state.messages[:-1]

        for chunk in stream_ollama_response(prompt_humanMessage, history_for_llm, system_prompt, model_sel,
                                            temperature):
            full_reply += chunk
            # Il simbolo '▌' simula il cursore che lampeggia durante la digitazione
            stream_placeholder.markdown(full_reply + "▌")

        stream_placeholder.markdown(full_reply)

    # Salva la risposta dell'assistente nello stato
    st.session_state.messages.append({"role": "assistant", "content": full_reply})

else:
    if not st.session_state.images:
        st.error("Inserire Immagine in input.", icon="🚨")
    if prompt_to_send is not None and prompt_to_send != "" :
        chunk_list=create_chunking(prompt_to_send)
        if len(chunk_list) > 1:
            st.error("Query troppo lunga")