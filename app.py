import streamlit as st

# ── Configurazione Pagina ─────────────────────────────
st.set_page_config(
    page_title="AI Studio",
    page_icon="✨", # Un'icona più moderna
    layout="wide",
    initial_sidebar_state="collapsed", # Più pulito per una landing page
)

# ── CSS Globale ─────────────────────────────
st.markdown("""
<style>
/* Importiamo font moderni per UI e Titoli */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Outfit:wght@700&display=swap');

/* Variabili per il tema chiaro moderno */
:root {
  --text-main: #111827;
  --text-muted: #6B7280;
  --bg-light: #F9FAFB;
  --card-bg: #FFFFFF;
  --border-light: #E5E7EB;
  --accent-gradient: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
}

/* Reset font di base usando Inter */
html, body, [class*="css"] {
  font-family: 'Inter', sans-serif !important;
}

/* ── Sezione Hero (Intestazione) ── */
.hero-section {
  text-align: center;
  padding: 4rem 1rem 3rem 1rem;
  animation: fadeIn 0.8s ease-out;
}

.main-title {
  font-family: 'Outfit', sans-serif;
  font-size: 5.5rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  line-height: 1.1;
  color: var(--text-main);
  margin-bottom: 0.5rem;
}

/* Effetto sfumato elegante per la parola STUDIO */
.main-title span {
  background: var(--accent-gradient);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.subtitle {
  color: var(--text-muted);
  font-size: 1.1rem;
  font-weight: 500;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  margin-top: 1rem;
}

/* ── Stile per i Container (Card) ── */
/* Intercettiamo il wrapper dei container di Streamlit per renderli moderni */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 16px !important;
    border: 1px solid var(--border-light) !important;
    background-color: var(--card-bg) !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    padding: 0.5rem !important; /* Respiro interno extra */
}

div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    transform: translateY(-4px) !important;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08), 0 4px 6px -2px rgba(0, 0, 0, 0.04) !important;
}

/* ── Bottoni Primari Personalizzati ── */
div[data-testid="stButton"] button[kind="primary"] {
    background: var(--accent-gradient) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1rem !important;
    transition: opacity 0.2s, transform 0.1s !important;
}

div[data-testid="stButton"] button[kind="primary"]:hover {
    opacity: 0.9 !important;
}

div[data-testid="stButton"] button[kind="primary"]:active {
    transform: scale(0.98) !important;
}

/* Animazione di entrata */
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(15px); }
  to { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)

# ── Header Hero ─────────────────────────────
st.markdown("""
<div class="hero-section">
  <div class="main-title">CLINICAL <span>RAG</span></div>
  <div class="subtitle">Sistema RAG avanzato in ambito clinico · Qdrant · Ollama · LangChain</div>
</div>
""", unsafe_allow_html=True)

# Spaziatura invisibile per dare respiro prima delle card
st.write("")
st.write("")

# ── Navigazione ─────────────────────────────
# Usiamo 3 colonne per centrare elegantemente il contenuto
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    with st.container(border=True):
        # Usiamo HTML interno per gestire i margini del testo in modo preciso
        st.markdown("""
        <div style="padding: 1rem 0.5rem;">
            <h3 style="font-family:'Outfit', sans-serif; color:#111827; margin-bottom: 0.5rem; margin-top: 0; font-size: 1.8rem; font-weight: 700;">
                <span style="font-size: 1.4rem; vertical-align: middle;">🩺</span> Chat Clinica Multimodale
            </h3>
            <p style="color:#6B7280; font-size: 1rem; line-height: 1.5; margin-bottom: 1.5rem;">
                Analizza radiografie del torace e referti clinici interagendo con modelli locali tramite Ollama. Ricerca vettoriale potenziata da Qdrant, BiomedCLIP e MedCPTCrossEncoder.
            </p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Avvia Conversazione", type="primary", use_container_width=True):
            st.switch_page("pages/1_multimodalChat_LLM.py")