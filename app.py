import logging
import math
import os
import numpy
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

# ─── Logging Configuration ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────
MAX_RESULTS = 3

# Load environment variables
load_dotenv()


def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    """Calculate cosine similarity between two vectors.

    Cosine similarity measures the angle between two vectors, producing a
    value between -1 and 1 where 1 means identical direction.

    Args:
        vector_a: First vector.
        vector_b: Second vector (must be the same length as vector_a).

    Returns:
        Cosine similarity score between -1 and 1.

    Raises:
        ValueError: If vectors have different dimensions.
    """
    if len(vector_a) != len(vector_b):
        raise ValueError("Vectors must have the same dimensions")

    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))

    return dot_product / (norm_a * norm_b)


def search_sentences(
    vector_store: InMemoryVectorStore,
    query: str,
    k: int = MAX_RESULTS,
) -> list[tuple[str, float]]:
    """Search for sentences similar to the given query in the vector store.

    Args:
        vector_store: The in-memory vector store to search against.
        query: The search query string to find similar sentences for.
        k: Maximum number of results to return. Defaults to MAX_RESULTS.

    Returns:
        A list of tuples containing (document_text, similarity_score),
        sorted by descending similarity.
    """
    results = vector_store.similarity_search_with_score(query, k=k)

    matched: list[tuple[str, float]] = []
    print(f'\n🔍 Search Results for "{query}":\n')
    for rank, (document, score) in enumerate(results, 1):
        print(f"{rank}. [Score: {score:.4f}] {document.page_content}")
        matched.append((document.page_content, score))

    return matched


def main() -> None:
    """Run the embedding inspector lab application."""
    logger.info("Python LangChain Agent Starting...")
    print("🤖 Python LangChain Agent Starting...\n")

    # Check for GitHub token
    if not os.getenv("GITHUB_TOKEN"):
        logger.error("GITHUB_TOKEN not found in environment variables")
        print("❌ Error: GITHUB_TOKEN not found in environment variables.")
        print("Please create a .env file with your GitHub token:")
        print("GITHUB_TOKEN=your-github-token-here")
        print("\nGet your token from: https://github.com/settings/tokens")
        print("Or use GitHub Models: https://github.com/marketplace/models")
        return

    # Create embeddings instance using GitHub Models API
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        base_url="https://models.inference.ai.azure.com",
        api_key=os.getenv("GITHUB_TOKEN"),
        check_embedding_ctx_length=False,
    )

    # Initialize the in-memory vector store with the embeddings model
    vector_store = InMemoryVectorStore(embeddings)

    # Test sentences for embedding comparison
    sentences = [
        "The canine barked loudly.",
        "The dog made a noise.",
        "The electron spins rapidly.",
        "I love eating pizza with extra cheese.",
        "The basketball player scored a three-pointer.",
        "Rain is forecasted for tomorrow afternoon.",
        "Python is a popular programming language.",
        "The kitten purred softly on the couch.",
        "Quantum mechanics explains particle behavior.",
        "Homemade pasta tastes better than store-bought.",
        "The soccer match ended in a tie.",
        "Clouds are forming over the mountains.",
        "JavaScript runs in web browsers.",
        "Puppies need lots of attention and exercise.",
        "Atoms are made of protons, neutrons, and electrons.",
    ]

    # Store sentences in the vector database with metadata
    print("=== Vector Store Lab ===")
    print(f"Storing {len(sentences)} sentences in the vector database...")

    metadatas = [
        {"created_at": datetime.now().isoformat(), "index": i}
        for i in range(len(sentences))
    ]
    vector_store.add_texts(texts=sentences, metadatas=metadatas)

    print(f"✅ Successfully stored {len(sentences)} sentences\n")
    for i, sentence in enumerate(sentences, 1):
        print(f"  {i}. {sentence}")

    # ── Interactive Search Loop ───────────────────────────────────────────
    print("\n=== Semantic Search ===")
    while True:
        query = input("\nEnter a search query (or 'quit' to exit): ").strip()

        if query.lower() in ("quit", "exit"):
            break

        if not query:
            continue

        search_sentences(vector_store, query)
        print()

    print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()