import logging
from abc import ABC, abstractmethod

import torch
from PIL import Image

from src.phaseOne.Embedding import TextEmbedding, ImageEmbedding
from src.phaseOne.chunking import create_chunking
from src.phaseOne.dataset import TxtDataset, ImageDataset
from src.phaseOne.model import load_model, DEVICE

device = "mps" if torch.cuda.is_available() else "cpu"

class CreateEmbeddings(ABC):

    @abstractmethod
    def create_embeddings(self):
        """
        Metodo astratto che deve essere implementato da tutte le classi figlie.
        Restituisce gli elementi del dataset.
        """
        pass

#TODO
class TextCreateEmbeddings(CreateEmbeddings):
    def __init__(self, text_dataset: TxtDataset):
        self.txt_dataset = text_dataset


    def create_embeddings(self) -> list[TextEmbedding]:
        model, _, tokenizer = load_model()
        model.to(DEVICE)
        model.eval()

        ret_embedding=[]

        txt_dataset= self.txt_dataset.get_items()
        for txt_file in txt_dataset:
            txt_path=self.txt_dataset.get_directory_path()/txt_file
            with open(txt_path, "r", encoding="utf-8") as f:
                string = f.read()

            chunks= create_chunking(string)
            for chunk in chunks:
                embedding = create_text_embedding(chunk,model,tokenizer,192).tolist()[0]
                text_embedding = TextEmbedding(txt_file,embedding,chunk)
                ret_embedding.append(text_embedding)

        return ret_embedding

#TODO
class ImageCreateEmbeddings(CreateEmbeddings):
    def __init__(self, img_dataset: ImageDataset):
        self.img_dataset = img_dataset

    def create_embeddings(self)-> list[ImageEmbedding]:
        model, preprocess, _ = load_model()
        model.to(DEVICE)
        model.eval()

        ret_embedding=[]
        for img_file in self.img_dataset.get_items():
            img_path=self.img_dataset.get_directory_path()/img_file
            img=Image.open(img_path).convert('RGB')
            embedding= create_image_embedding(img,model,preprocess).tolist()[0]
            img_embedding = ImageEmbedding(img_file,embedding)
            ret_embedding.append(img_embedding)

        return ret_embedding


def create_image_embedding(image: Image.Image, model, preprocess) -> torch.Tensor:

    image_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        image_features = model.encode_image(image_tensor)

    img_emb = image_features / image_features.norm(dim=-1, keepdim=True)

    return img_emb



def create_text_embedding(chunk:str, model, tokenizer, context_length=256)-> torch.Tensor:


    text_tokens = tokenizer([chunk], context_length=context_length).to(DEVICE)


    with torch.no_grad():
        text_features = model.encode_text(text_tokens)

    txt_emb = text_features / text_features.norm(dim=-1, keepdim=True)

    return txt_emb


def create_query_embedding(query:str)-> TextEmbedding:

    model, _, tokenizer = load_model()
    model.to(DEVICE)
    model.eval()

    embedding= create_text_embedding(query,model,tokenizer).tolist()[0]
    text_embedding = TextEmbedding(None,embedding,query)
    return text_embedding


def create_image_query_embedding(img:Image.Image)-> ImageEmbedding:
    model, preprocess, _ = load_model()
    model.to(DEVICE)
    model.eval()

    embedding=create_image_embedding(img,model, preprocess).tolist()[0]
    img_embedding = ImageEmbedding(None,embedding)
    return img_embedding
