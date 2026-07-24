import os
from abc import ABC, abstractmethod
import csv

from paths import TXT_DIR


class Dataset(ABC):

    @abstractmethod
    def get_items(self):
        """
        Metodo astratto che deve essere implementato da tutte le classi figlie.
        Restituisce gli elementi del dataset.
        """
        pass


class TextCSVDataset(Dataset):

    def __init__(self, filepath):
        self.__filepath = filepath
        self.__data = {}

        try:
            with open(self.__filepath, mode='r', encoding='utf-8') as file:
                reader = csv.DictReader(file)

                for row in reader:
                    print(row)
                    id_ = row.get("id")
                    value = row.get("text")

                    if id_ and value:
                        self.__data[id_] = value

        except FileNotFoundError:
            raise FileNotFoundError(f"File non trovato: {self.__filepath}")

    def get_items(self)->dict:
        return self.__data.copy()


class ImageDataset(Dataset):
    def __init__(self, directorypath):
        self.__directorypath = directorypath
        self.__images = []

    def get_items(self)->list:
        estensioni_immagini = ('.png', '.jpg', '.jpeg', '.gif')
        try:
            files = os.listdir(self.__directorypath)
            self.__images= [f for f in files if f.lower().endswith(estensioni_immagini)]
            return self.__images
        except FileNotFoundError:
            print("Cartella non trovata.")
            return []
        except Exception as e:
            print(f"Errore: {e}")
            return []
    def get_directory_path(self)->str:
        return self.__directorypath

class TxtDataset(Dataset):
    def __init__(self, directorypath):
        self.__directorypath = directorypath
        self.__text = []

    def get_items(self)->list:
        try:
            files = os.listdir(self.__directorypath)
            self.__text= [f for f in files if f.lower().endswith('.txt')]
            return self.__text
        except FileNotFoundError:
            print("Cartella non trovata.")
            return []
        except Exception as e:
            print(f"Errore: {e}")
            return []

    def get_directory_path(self):
        return self.__directorypath