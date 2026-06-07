from backend.services.LLM_service.embedding_service import get_embedding_service

def get_embedding_model():
    """
    Exposes the centralized LLMEmbeddingService as a model-like object with .encode()
    """
    return get_embedding_service()

def get_embeddings(texts):
    """
    Retrieves embeddings using the centralized service.
    """
    return get_embedding_service().get_embeddings(texts)
