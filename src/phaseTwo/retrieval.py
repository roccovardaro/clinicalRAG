import io
import os
import warnings

from PIL import Image
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from qdrant_client import QdrantClient
from qdrant_client.conversions.common_types import ScoredPoint

from paths import IMG_DIR, TXT_DIR
from src.phaseOne.createEmbeddings import create_query_embedding, create_image_query_embedding
from src.phaseTwo.prompt_llm import create_human_prompt
from src.phaseTwo.reranking import reranking
import logging
from transformers.utils import logging as hf_logging

# =========================
# SILENZIA WARNING
# =========================
warnings.filterwarnings("ignore")

# =========================
# SILENZIA HUGGINGFACE
# =========================
hf_logging.set_verbosity_error()

# =========================
# LOGGER RUMOROSI
# =========================
for noisy_logger in [
    "transformers",
    "huggingface_hub",
    "PIL",
    "torch",
    "urllib3",
    "accelerate",
]:
    log = logging.getLogger(noisy_logger)
    log.setLevel(logging.ERROR)
    log.propagate = False
    log.handlers.clear()

# =========================
# TUO LOGGER PERSONALIZZATO
# =========================
GREEN = "\033[92m"
RESET = "\033[0m"

formatter = logging.Formatter(
    GREEN + "%(asctime)s | %(levelname)s | %(name)s | %(message)s" + RESET
)

handler = logging.StreamHandler()
handler.setFormatter(formatter)

logger = logging.getLogger("retrieval_pipeline")

# evita duplicazioni
logger.handlers.clear()

logger.addHandler(handler)
logger.setLevel(logging.INFO)

# IMPORTANTISSIMO
logger.propagate = False

load_dotenv()
HOST_QDRANT=os.getenv('HOST_QDRANT')
PORT_QDRANT=os.getenv('PORT_QDRANT')


def query_embedding(query:str):
    query_vector = create_query_embedding(query).get_embedding()
    return query_vector

def image_embedding(query:Image.Image):
    query_vector = create_image_query_embedding(query).get_embedding()
    return query_vector

def search_top_k_cosine_similarity_from_text(query:str, collection_name:str, k:int=30) -> list[ScoredPoint]:
    query_vector = query_embedding(query)
    client = QdrantClient(HOST_QDRANT, port=int(PORT_QDRANT))

    search_result = client.query_points(
        collection_name= collection_name,
        query=query_vector,
        with_payload=True,
        limit=k
    ).points
    return search_result

def search_top_k_cosine_similarity_from_image(query:Image.Image, collection_name:str, k:int=30) -> list[ScoredPoint]:
    query_vector = image_embedding(query)
    client = QdrantClient(HOST_QDRANT, port=int(PORT_QDRANT))

    search_result = client.query_points(
        collection_name= collection_name,
        query=query_vector,
        with_payload=True,
        limit=k
    ).points
    return search_result


def merge_scoredPoint_lists(list1: list[ScoredPoint], list2: list[ScoredPoint]) -> list[ScoredPoint]:
    seen_ids = set()
    result = []

    for point in list1 + list2:
        if point.id not in seen_ids:
            seen_ids.add(point.id)
            result.append(point)

    return result



def extract_payload_from_scoredPoint_list(scored_points:list[ScoredPoint])->list[dict]:

    result_list = []
    for point in scored_points:
        payload=point.payload
        result_list.append(payload)
    return result_list


def get_full_text_from_payload(text_payload:list[dict]) -> list[str]:
    result_list = []
    for text in text_payload:
        text_path= text['id']
        text_path_absolute= TXT_DIR / text_path

        with open(text_path_absolute, "r", encoding="utf-8") as f:
            full_text = f.read()

        result_list.append(full_text)

    return result_list


def get_text_to_LLM(query:str,text_payload:list[dict],top_k)-> list[str]:

    full_text=get_full_text_from_payload(text_payload)
    text_result_list=reranking(query, full_text, top_k)
    return text_result_list


def get_image_to_LLM(image_payload:list[dict])-> list[str]:
    """

    :param image_payload:
    :return: lista con i path assoluti delle immagini
    """
    result_list=[]
    for image in image_payload:
        image_path= image['id']
        img_path_absolute = IMG_DIR / image_path
        result_list.append(img_path_absolute)
    return result_list




def retrival_to_LLM(query: str, query_image:bytes,top_k=12) -> HumanMessage:

    image = Image.open(io.BytesIO(query_image))

    logger.info(f"[retrieval] Query ricevuta: {query}")
    logger.info(f"[retrieval] top_k iniziale: {top_k}")

    # CALCOLO TOP_K_SIMILARITY
    TEXT_COLLECTION = os.getenv("TEXT_COLLECTION_NAME")
    IMAGE_COLLECTION = os.getenv("IMAGE_COLLECTION_NAME")

    top_k_text_from_text = search_top_k_cosine_similarity_from_text(query, TEXT_COLLECTION, top_k)
    #logger.info(f"[retrieval] Risultati Testo dal Testo recuperati: {top_k_text_from_text}")
    logger.info(f"[retrieval] lunghezza Testo da Testo recuperati: {len(top_k_text_from_text)} ")

    top_k_image_from_text = search_top_k_cosine_similarity_from_text(query, IMAGE_COLLECTION, top_k // 3)
    #logger.info(f"[retrieval] Risultati Immagine dal Testo recuperati: {top_k_image_from_text}")
    logger.info(f"[retrieval] lunghezza Immagini da Testo recuperati: {len(top_k_image_from_text)} ")

    top_k_text_from_image= search_top_k_cosine_similarity_from_image(image, TEXT_COLLECTION,top_k //3)
    #logger.info(f"[retrieval] Risultati Testo dalle Immagini recuperati: {top_k_text_from_image}")
    logger.info(f"[retrieval] lunghezza Testo da Immagini recuperati: {len(top_k_text_from_image)} ")

    top_k_image_from_image= search_top_k_cosine_similarity_from_image(image, IMAGE_COLLECTION,top_k)
    #logger.info(f"[retrieval] Risultati Immagini dal Testo recuperati: {top_k_image_from_image}")
    logger.info(f"[retrieval] lunghezza Immagini da Immagini recuperati: {len(top_k_image_from_image)} ")

    top_k_text= merge_scoredPoint_lists(top_k_text_from_text, top_k_text_from_image)
    #logger.info(f"[retrival] Risultato lista testi merge: {top_k_text}")
    logger.info(f"[retrieval] lunghezza lista testi merge: {len(top_k_text)} ")

    logger.info(f"[retrieval]top k testi merge")
    for t in top_k_text:
        logger.info(f"{t.id} : {t.score}")

    top_k_image= merge_scoredPoint_lists(top_k_image_from_text, top_k_image_from_image)
    #logger.info(f"[retrival] Risultato lista immagini merge: {top_k_image}")
    logger.info(f"[retrieval] lunghezza lista immagini merge: {len(top_k_image)} ")

    logger.info(f"[retrieval]top k immagini merge")
    for t in top_k_image:
        logger.info(f"{t.id} : {t.score}")

    # RECUPERO IMMAGINI
    image_payload = extract_payload_from_scoredPoint_list(top_k_image)
    logger.debug(f"[retrieval] image_payload size: {len(image_payload)}")

    image_list = get_image_to_LLM(image_payload)
    logger.info(f"[retrieval] immagini pronte per LLM: {len(image_list)}")

    # RECUPERO TESTO
    text_payload = extract_payload_from_scoredPoint_list(top_k_text)
    logger.debug(f"[retrieval] text_payload size: {len(text_payload)}")

    text_list = get_text_to_LLM(query, text_payload,top_k // 2)
    logger.info(f"[retrieval] testi pronti per LLM: {len(text_list)}")

    # CREAZIONE HUMAN PROMPT
    human_prompt = create_human_prompt(query, query_image, text_list, image_list)
    logger.info(f"[retrieval] Human prompt creato")

    return human_prompt