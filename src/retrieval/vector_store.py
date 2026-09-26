from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from src.config.settings import VECTORSTORE_PATH
from src.retrieval.embeddings import embed_documents, embed_query
from src.tools.structured_query import entity_values


COLLECTION_NAME = "cosmetic_chemicals_v1"


def get_vector_store() -> QdrantClient:
    Path(VECTORSTORE_PATH).parent.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(VECTORSTORE_PATH))


def build_chemical_index(force: bool = False) -> int:
    store = get_vector_store()
    try:
        collections = {item.name for item in store.get_collections().collections}
        if COLLECTION_NAME in collections:
            current_count = store.count(COLLECTION_NAME).count
            if current_count and not force:
                return current_count
            store.delete_collection(COLLECTION_NAME)

        names = entity_values("chemical")
        vectors = embed_documents(
            [f"Reported cosmetic chemical: {name}" for name in names]
        )
        store.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=len(vectors[0]), distance=models.Distance.COSINE),
        )
        points = [
            models.PointStruct(
                id=str(uuid5(NAMESPACE_URL, f"cosmetic-chemical:{name.casefold()}")),
                vector=vector,
                payload={"entity_type": "chemical", "chemical_name": name},
            )
            for name, vector in zip(names, vectors)
        ]
        store.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
        return store.count(COLLECTION_NAME).count
    finally:
        store.close()


def search_chemical_vectors(query: str, limit: int = 5) -> list[dict[str, object]]:
    if not Path(VECTORSTORE_PATH).exists():
        raise FileNotFoundError("Chemical vector index is missing; run scripts/build_vector_index.py first.")
    store = get_vector_store()
    try:
        if COLLECTION_NAME not in {item.name for item in store.get_collections().collections}:
            raise FileNotFoundError("Chemical vector index is missing; run scripts/build_vector_index.py first.")
        vector = embed_query(query)
        matches = store.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            limit=limit,
            with_payload=True,
        ).points
        return [
            {
                "chemical_name": point.payload["chemical_name"],
                "score": round(float(point.score), 4),
            }
            for point in matches
        ]
    finally:
        store.close()