import base64
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_ollama import ChatOllama
from src.phaseOne.chunking import create_chunking

load_dotenv()
HOST_OLLAMA=os.getenv('HOST_OLLAMA')
PORT_OLLAMA=os.getenv('PORT_OLLAMA')

# leggi immagine e converti in base64
def encode_image(img:bytes):
    return base64.b64encode(img).decode("utf-8")


def create_system_prompt() -> SystemMessage:
    system_message = SystemMessage(
        content="""
    You are a multimodal QA assistant.

    Task:
    - You must answer the user's question.
    - The user's main question is provided in the "USER QUERY" section.
    - The main image associated with the question is provided in the "PRIMARY USER IMAGE" section.
    - The subsequent texts and images are supporting content that can be used as additional context.

    Mandatory rules:
    - Exclusively use the information present in the provided inputs.
    - Do not use external knowledge.
    - Do not invent information.
    - Do not infer details that are not explicitly supported.
    - Answer ONLY the USER QUERY and the PRIMARY USER IMAGE.
    - Use supporting content only if it is useful to answer.
    - Do not cite, mention, or describe the sources.
    - Do not refer to the existence of images, texts, documents, or context.
    - Do not use phrasing such as:
      - "in the image"
      - "from the text"
      - "from the sources"
      - "from the document"
      - "is shown"
      - "is seen"
      - "the provided context"
    - Answer the final question directly.
    - Keep the answer precise and factual.
    """
    )

    return system_message


def create_human_prompt(
    query: str,
    query_image: bytes,
    text_list: list[str],
    img_path_list: list[str]
    )-> HumanMessage:

    human_content = [
        {
            "type": "text",
            "text": f"""
================ USER QUERY ================

{query}
"""
        }
    ]

    # immagine principale dell'utente
    if query_image is not None:

        human_content.append({
            "type": "text",
            "text": """
================ PRIMARY USER IMAGE ================

Questa è l'immagine principale associata direttamente
alla query dell'utente.
"""
        })

        human_content.append({
            "type": "image_url",
            "image_url": encode_image(query_image)
        })

    # testi di supporto
    if text_list:

        human_content.append({
            "type": "text",
            "text": """
================ SUPPORTING TEXTS ================

I seguenti testi sono contenuti di supporto
utilizzabili per rispondere alla query.
"""
        })

        for i, text in enumerate(text_list, start=1):

            human_content.append({
                "type": "text",
                "text": f"""
--- SUPPORT TEXT {i} ---

{text}
"""
            })

    # immagini di supporto
    if img_path_list:

        human_content.append({
            "type": "text",
            "text": """
================ SUPPORTING IMAGES ================

Le seguenti immagini sono contenuti di supporto
utilizzabili per rispondere alla query.
"""
        })

        for i, img_path in enumerate(img_path_list, start=1):

            with open(img_path, "rb") as f:
                img = f.read()

            img_base64 = encode_image(img)

            human_content.append({
                "type": "text",
                "text": f"""
--- SUPPORT IMAGE {i} ---
"""
            })

            human_content.append({
                "type": "image_url",
                "image_url": f"data:image/png;base64,{img_base64}"
            })

    human_message = HumanMessage(content=human_content)

    return human_message


def prompt_is_ok(text: str) -> bool:
    if text is None:
        return False

    text = text.strip()

    if not text:
        return False

    chunks = create_chunking(text)

    return len(chunks) == 1


def stream_ollama_response(prompt: HumanMessage, history: list, system: SystemMessage, model: str, temperature: float):

    llm = ChatOllama(
        model=model,
        temperature=temperature,
        streaming=True,
        base_url=f"http://{HOST_OLLAMA}:{PORT_OLLAMA}",
    )

    lc_messages = [system]

    # Ricostruzione della cronologia
    for msg in history:
        if msg["role"] == "user":
            if msg.get("image"):
                # Messaggio multimodale storico
                lc_messages.append(HumanMessage(content=[
                    {"type": "text", "text": msg["content"]},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{msg['image']}"}}
                ]))
            else:
                lc_messages.append(HumanMessage(content=msg["content"]))
        else:
            lc_messages.append(AIMessage(content=msg["content"]))

    lc_messages.append(prompt)


    try:
        for chunk in llm.stream(lc_messages):
            if chunk.content:
                yield chunk.content
    except Exception as e:
        yield f"\n\n⚠️ Errore di connessione a Ollama: {e}"


