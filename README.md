# ClinicalRAG

ClinicalRAG è un sistema avanzato di Retrieval-Augmented Generation (RAG) in ambito clinico. Sviluppato per interagire sia con testo che con immagini, il progetto si avvale di modelli locali tramite Ollama, LangChain e Qdrant (Vector Database) per offrire un'interfaccia chat multimodale veloce e sicura tramite Streamlit.

---

## 🏗 Struttura del Progetto

Il progetto è modulare e diviso logicamente in due fasi (creazione embeddings/indicizzazione e retrieval/generazione):

```text
ClinicalRAG/
├── app.py
├── pages/
│   └── 1_multimodalChat_LLM.py
├── src/
│   ├── phaseOne/
│   │   ├── chunking.py
│   │   ├── Embedding.py
│   │   ├── createEmbeddings.py
│   │   ├── dataset.py
│   │   ├── model.py
│   │   └── qdrantHandler.py
│   └── phaseTwo/
│       ├── model.py
│       ├── prompt_llm.py
│       ├── reranking.py
│       └── retrieval.py
├── dataset/
│   ├── images/
│   ├── txt/
│   └── output.csv
├── models/
│   ├── biomedCLIP/
│   └── medcptcrossencoder/
├── test/
├── paths.py
├── download_models.sh
├── requirements.txt
└── .env
```

---

## 🛠 Requisiti e Tecnologie

Le principali tecnologie utilizzate all'interno del progetto sono:
- **Frontend / UI:** [Streamlit](https://streamlit.io/)
- **Gestione LLM locale:** [Ollama](https://ollama.com/)
- **Orchestrazione:** [LangChain](https://www.langchain.com/) / LangGraph
- **Vector Database:** [Qdrant](https://qdrant.tech/)
- **Machine Learning & NLP:** PyTorch, Transformers (HuggingFace), BiomedCLIP, MedCPTCrossEncoder

---

## 📊 Dataset

Il dataset utilizzato in questo progetto contiene radiografie del torace e relativi report. Puoi scaricarlo dal seguente link su Kaggle:
- [IU Chest X-Rays (Cleaned)](https://www.kaggle.com/datasets/masrursabab/iu-chest-x-rays-cleaned)

Una volta scaricato e compresso l'archivio, dovrai posizionare i file all'interno della cartella `dataset/` presente nella root del progetto, rispettando scrupolosamente questa struttura:

```text
ClinicalRAG/
└── dataset/
    ├── images/       # Inserisci qui tutte le immagini delle radiografie
    ├── txt/          # Inserisci qui tutti i file di testo (.txt) contenenti i referti clinici
```

Assicurati che i file siano esattamente in questi percorsi, poiché gli script di indicizzazione (Fase 1) cercheranno i dati proprio in queste cartelle.

---

## ⚙️ Configurazione Ambiente (`.env`)

Prima di avviare il progetto, è necessario creare un file `.env` nella root del progetto e definire le variabili d'ambiente richieste per le connessioni ai servizi. 

Ecco un esempio di configurazione (puoi copiare questo contenuto nel tuo file `.env`):

```env
# Configurazione Qdrant
HOST_QDRANT=localhost
PORT_QDRANT=6333
TEXT_COLLECTION_NAME="TEXT_COLLECTION"
IMAGE_COLLECTION_NAME="IMAGE_COLLECTION"

# Configurazione Ollama
HOST_OLLAMA=localhost
PORT_OLLAMA=11434
```

- **`HOST_QDRANT` / `PORT_QDRANT`**: L'indirizzo e la porta del database vettoriale Qdrant (di default `localhost` e `6333`).
- **`TEXT_COLLECTION_NAME` / `IMAGE_COLLECTION_NAME`**: I nomi delle collezioni su Qdrant in cui salvare e recuperare gli embeddings per testi e immagini.
- **`HOST_OLLAMA` / `PORT_OLLAMA`**: L'indirizzo e la porta del server locale Ollama (di default `localhost` e `11434`).

---

## 🚀 Come avviare il progetto

Per avviare il progetto in locale (Ambiente Virtuale Python):

1. **Clona il repository (se non lo hai già fatto) e spostati nella cartella del progetto:**
   ```bash
   cd ClinicalRAG
   ```

2. **Crea e attiva un ambiente virtuale (consigliato):**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Su Windows: .venv\Scripts\activate
   ```

3. **Installa le dipendenze richieste:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Scarica i modelli necessari:**
   Esegui l'apposito script per scaricare i modelli in locale (es. BiomedCLIP e MedCPTCrossEncoder). Assicurati di avere l'ambiente virtuale attivato:
   ```bash
   bash download_models.sh
   ```

5. **Avvia Qdrant (Database Vettoriale):**
   Il sistema richiede Qdrant per il salvataggio e il recupero degli embeddings. Assicurati di avere un'istanza di Qdrant in esecuzione (ad esempio in locale o tramite un servizio gestito) all'indirizzo e porta configurati nel file `.env`.

6. **Avvia Ollama in locale:**
   Assicurati che l'applicazione Ollama sia in esecuzione sul tuo PC e che i modelli necessari siano scaricati (es. Gemma4 o altri modelli multimodali).

7. **Avvia l'applicazione Streamlit:**
   ```bash
   streamlit run app.py
   ```

8. **Accedi all'interfaccia:** 
   Il browser si aprirà automaticamente (solitamente all'indirizzo `http://localhost:8501`).

---

## 🧠 Flusso di Lavoro del Sistema (RAG)

Il progetto è diviso in due macro fasi. Di seguito viene illustrata l'architettura generale dell'interfaccia utente:



### Fase 1 (Indicizzazione)
I documenti clinici (testi e/o immagini) vengono caricati, puliti e suddivisi in chunk più piccoli (`chunking.py`). Successivamente vengono generati gli embeddings testuali e/o visivi (utilizzando `BiomedCLIP`) e salvati all'interno del database vettoriale Qdrant (`qdrantHandler.py`).

![Architettura Fase 1 - Indicizzazione](images/indexing_architecture.png)

### Fase 2 (Retrieval e Generazione)
L'utente interagisce tramite l'interfaccia chat. La query dell'utente viene vettorializzata e utilizzata per recuperare i documenti più rilevanti dal Vector DB (`retrieval.py`). I risultati grezzi subiscono un reranking tramite `MedCPTCrossEncoder` (`reranking.py`) per migliorarne la pertinenza. Infine, i documenti e le immagini selezionate vengono passati al LLM locale tramite Ollama per generare una risposta precisa e contestualizzata (`prompt_llm.py`).

![Architettura Fase 2 - Retrieval e Generazione](images/retrieval_architecture.png)

![Architettura GUI](images/gui_architecture.png)
---
