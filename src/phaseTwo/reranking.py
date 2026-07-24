import torch

from src.phaseTwo.model import load_model


def reranking_text(query:str,docs:list[str])->dict:


    dict_ret={} #{key:score , value: text}

    pairs = [[query, doc] for doc in docs]

    model, tokenizer= load_model()

    with torch.no_grad():
        encoded = tokenizer(
            pairs,
            truncation=True,
            padding=True,
            return_tensors="pt",
            max_length=512,
        )

        logits = model(**encoded).logits.squeeze(dim=1)

    for i in range(len(docs)):
        dict_ret[logits[i]]=docs[i]

    return dict_ret



def reranking(query:str,docs:list[str], top_k:int):

    list_ret=[]
    score_text_dict= reranking_text(query,docs)

    score_sorted_top_k=sorted(score_text_dict.keys())[:top_k]

    for score in score_sorted_top_k:
        list_ret.append(score_text_dict[score])
    return list_ret


