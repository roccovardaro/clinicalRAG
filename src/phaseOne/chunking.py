import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

def splitter_text(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=192,
        chunk_overlap=30,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_text(text)


def create_chunking(text:str):
    return splitter_text(text)

