import logging
import math
import os
from datetime import datetime
from typing import Optional

import numpy as np
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

# Type alias for either vector store backend
VectorStore = InMemoryVectorStore | Chroma

# ─── Logging Configuration ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────
MAX_RESULTS = 3
SIMILARITY_THRESHOLD = 0.7
CATEGORIES = ["animals", "science", "food", "sports", "weather", "technology"]
CHROMA_PERSIST_DIR = "./chroma_db"

# Sentence-to-category mapping for metadata filtering
SENTENCE_CATEGORIES: dict[str, str] = {
    "The canine barked loudly.": "animals",
    "The dog made a noise.": "animals",
    "The electron spins rapidly.": "science",
    "I love eating pizza with extra cheese.": "food",
    "The basketball player scored a three-pointer.": "sports",
    "Rain is forecasted for tomorrow afternoon.": "weather",
    "Python is a popular programming language.": "technology",
    "The kitten purred softly on the couch.": "animals",
    "Quantum mechanics explains particle behavior.": "science",
    "Homemade pasta tastes better than store-bought.": "food",
    "The soccer match ended in a tie.": "sports",
    "Clouds are forming over the mountains.": "weather",
    "JavaScript runs in web browsers.": "technology",
    "Puppies need lots of attention and exercise.": "animals",
    "Atoms are made of protons, neutrons, and electrons.": "science",
}

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

    a = np.array(vector_a)
    b = np.array(vector_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def search_sentences(
    vector_store: VectorStore,
    query: str,
    k: int = MAX_RESULTS,
    category: Optional[str] = None,
    threshold: Optional[float] = None,
) -> list[tuple[str, float]]:
    """Search for sentences similar to the given query in the vector store.

    Args:
        vector_store: The in-memory vector store to search against.
        query: The search query string to find similar sentences for.
        k: Maximum number of results to return. Defaults to MAX_RESULTS.
        category: Optional category filter to restrict results to a specific
            category (e.g., "animals", "science").
        threshold: Optional minimum similarity score. Results below this
            threshold are excluded. Defaults to None (no filtering).

    Returns:
        A list of tuples containing (document_text, similarity_score),
        sorted by descending similarity.
    """
    # ── Challenge 1: Metadata Filtering ───────────────────────────────────
    filter_dict: Optional[dict[str, str]] = None
    if category:
        filter_dict = {"category": category}

    results = vector_store.similarity_search_with_score(
        query, k=k, filter=filter_dict
    )

    # ── Challenge 3: Threshold Filtering ──────────────────────────────────
    matched: list[tuple[str, float]] = []
    label = f'"{query}"'
    if category:
        label += f" [category: {category}]"
    if threshold is not None:
        label += f" [threshold: {threshold}]"

    print(f"\n🔍 Search Results for {label}:\n")
    for rank, (document, score) in enumerate(results, 1):
        if threshold is not None and score < threshold:
            print(f"  {rank}. [Score: {score:.4f}] ❌ Below threshold — {document.page_content}")
            continue
        print(f"  {rank}. [Score: {score:.4f}] {document.page_content}")
        matched.append((document.page_content, score))

    if not matched:
        print("  No results matched the criteria.")

    return matched


def hybrid_search(
    vector_store: VectorStore,
    query: str,
    k: int = MAX_RESULTS,
    category: Optional[str] = None,
    threshold: Optional[float] = None,
) -> list[tuple[str, float]]:
    """Combine vector similarity search with keyword matching.

    Hybrid search boosts results that match both semantically and by
    keyword, giving higher rank to documents that contain the query words.

    Args:
        vector_store: The in-memory vector store to search against.
        query: The search query string.
        k: Maximum number of results to return. Defaults to MAX_RESULTS.
        category: Optional category filter for metadata filtering.
        threshold: Optional minimum similarity score for threshold filtering.

    Returns:
        A list of tuples containing (document_text, combined_score),
        sorted by descending combined score.
    """
    # ── Challenge 2: Hybrid Search ────────────────────────────────────────
    filter_dict: Optional[dict[str, str]] = None
    if category:
        filter_dict = {"category": category}

    results = vector_store.similarity_search_with_score(
        query, k=k, filter=filter_dict
    )

    # Keyword matching: check if query words appear in the document
    query_words = set(query.lower().split())
    scored: list[tuple[str, float, float, float]] = []

    for document, sim_score in results:
        doc_words = set(document.page_content.lower().split())
        # Jaccard similarity for keyword overlap
        keyword_score = (
            len(query_words & doc_words) / len(query_words | doc_words)
            if query_words | doc_words
            else 0.0
        )
        # Combined score: 70% semantic + 30% keyword
        combined_score = 0.7 * sim_score + 0.3 * keyword_score
        scored.append((document.page_content, sim_score, keyword_score, combined_score))

    # Sort by combined score descending
    scored.sort(key=lambda x: x[3], reverse=True)

    label = f'"{query}" (hybrid)'
    if category:
        label += f" [category: {category}]"
    if threshold is not None:
        label += f" [threshold: {threshold}]"

    print(f"\n🔍 Hybrid Search Results for {label}:\n")
    matched: list[tuple[str, float]] = []
    for rank, (text, sim, kw, combined) in enumerate(scored, 1):
        if threshold is not None and combined < threshold:
            print(
                f"  {rank}. [Combined: {combined:.4f} | Semantic: {sim:.4f} | "
                f"Keyword: {kw:.4f}] ❌ Below threshold — {text}"
            )
            continue
        print(
            f"  {rank}. [Combined: {combined:.4f} | Semantic: {sim:.4f} | "
            f"Keyword: {kw:.4f}] {text}"
        )
        matched.append((text, combined))

    if not matched:
        print("  No results matched the criteria.")

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

    # ── Challenge 4: Choose vector store backend ──────────────────────────
    print("Choose vector store backend:")
    print("  1. InMemoryVectorStore (default, data lost on exit)")
    print("  2. Chroma (persistent, data saved to disk)")
    choice = input("Enter 1 or 2 [1]: ").strip()

    if choice == "2":
        vector_store: VectorStore = Chroma(
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
            collection_name="lab_sentences",
        )
        store_type = "Chroma (persistent)"
        logger.info("Using Chroma vector store with persist_directory=%s", CHROMA_PERSIST_DIR)
    else:
        vector_store = InMemoryVectorStore(embeddings)
        store_type = "InMemory (ephemeral)"
        logger.info("Using InMemoryVectorStore")

    print(f"📦 Vector store: {store_type}\n")

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
    # ── Challenge 1: Category metadata for filtering ───────────────────────
    print("=== Vector Store Lab ===")
    print(f"Storing {len(sentences)} sentences in the vector database...")

    metadatas = [
        {
            "created_at": datetime.now().isoformat(),
            "index": i,
            "category": SENTENCE_CATEGORIES.get(sentence, "general"),
        }
        for i, sentence in enumerate(sentences)
    ]
    vector_store.add_texts(texts=sentences, metadatas=metadatas)

    print(f"✅ Successfully stored {len(sentences)} sentences\n")
    for i, sentence in enumerate(sentences, 1):
        cat = SENTENCE_CATEGORIES.get(sentence, "general")
        print(f"  {i}. [{cat}] {sentence}")

    # ── Interactive Search Loop ───────────────────────────────────────────
    print("\n=== Semantic Search ===")
    print("Available categories:", ", ".join(CATEGORIES))
    print("Commands: 'quit' to exit, 'mode' to toggle search mode")
    print("Options : 'cat <category>' to filter by category")
    print("          'threshold <value>' to set minimum score")
    print("          'mode' to toggle between semantic and hybrid search")

    # Search state
    search_mode = "semantic"  # "semantic" or "hybrid"
    active_category: Optional[str] = None
    active_threshold: Optional[float] = None

    while True:
        prompt = f"\n[{search_mode}] Enter query"
        if active_category:
            prompt += f" (cat: {active_category})"
        if active_threshold is not None:
            prompt += f" (threshold: {active_threshold})"
        prompt += " (or 'quit' to exit): "
        query = input(prompt).strip()

        if query.lower() in ("quit", "exit"):
            break

        if not query:
            continue

        # Toggle search mode
        if query.lower() == "mode":
            search_mode = "hybrid" if search_mode == "semantic" else "semantic"
            print(f"🔄 Switched to {search_mode} search mode.")
            continue

        # Set category filter
        if query.lower().startswith("cat "):
            requested_category = query[4:].strip().lower()
            if requested_category in CATEGORIES:
                active_category = requested_category
                print(f"📂 Category filter set to: {active_category}")
            elif requested_category == "all" or requested_category == "none":
                active_category = None
                print("📂 Category filter cleared.")
            else:
                print(f"❌ Unknown category '{requested_category}'.")
                print(f"   Available: {', '.join(CATEGORIES)}")
            continue

        # Set threshold filter
        if query.lower().startswith("threshold "):
            try:
                active_threshold = float(query[10:].strip())
                print(f"🎯 Similarity threshold set to: {active_threshold}")
            except ValueError:
                print("❌ Invalid threshold value. Use a number like 0.7")
            continue

        # Perform search
        if search_mode == "hybrid":
            hybrid_search(
                vector_store,
                query,
                category=active_category,
                threshold=active_threshold,
            )
        else:
            search_sentences(
                vector_store,
                query,
                category=active_category,
                threshold=active_threshold,
            )
        print()

    print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()