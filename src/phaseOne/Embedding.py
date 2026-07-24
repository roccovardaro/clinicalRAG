from abc import ABC, abstractmethod


class Embedding(ABC):

    @abstractmethod
    def get_embedding(self):
        pass


class TextEmbedding(Embedding):
    def __init__(self, id: str,embedding:list, text:str):
        self.id = id
        self.embedding = embedding
        self.text = text

    def get_embedding(self):
        return self.embedding

    def get_qdrant_embedding(self):
        payload = {"id":self.id,"text":self.text}
        return self.embedding, payload



class ImageEmbedding(Embedding):
    def __init__(self, id: str, embedding: list):
        self.id = id
        self.embedding = embedding

    def get_embedding(self):
        return self.embedding

    def get_qdrant_embedding(self):
        payload = {"id": self.id}
        return self.embedding, payload


