from langchain_ollama import OllamaEmbeddings

from src.config.settings import OLLAMA_BASE_URL, OLLAMA_EMBEDDING_MODEL


def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=OLLAMA_EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


def embed_documents(texts: list[str]) -> list[list[float]]:
    embeddings = get_embeddings()
    try:
        return embeddings.embed_documents(texts)
    finally:
        embeddings._client.close()


def embed_query(text: str) -> list[float]:
    embeddings = get_embeddings()
    try:
        return embeddings.embed_query(text)
    finally:
        embeddings._client.close()