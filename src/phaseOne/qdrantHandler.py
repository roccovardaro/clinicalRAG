import os
import uuid
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()
HOST_QDRANT=os.getenv('HOST_QDRANT')
PORT_QDRANT=os.getenv('PORT_QDRANT')


class QdrantHandler:
    def __init__(self,size:int,collection_name="MY_COLLECTION"):
        self.client = QdrantClient(url=f'''http://{HOST_QDRANT}:{PORT_QDRANT}''')
        self.size=size
        self.collection_name=collection_name
        if self.collection_name not in [c.name for c in self.client.get_collections().collections]:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.size, distance=Distance.COSINE),
            )

    def insert_vector(self, vector, payload: dict):
        try:
            points_structure=create_point_structure(vector,payload)
            operation_info = self.client.upsert(
                collection_name=self.collection_name,
                wait=True,
                points=[points_structure],
            )
        except Exception as e:
            print(e)
            return None

        return operation_info

def create_point_structure(vector,payload):

    point = PointStruct(
        id=str(uuid.uuid4()),
        vector=vector,
        payload=payload
    )

    return point



