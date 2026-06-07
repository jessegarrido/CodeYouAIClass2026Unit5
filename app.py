import logging
import os
from datetime import datetime
from typing import Optional, Union

import numpy as np
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

# Union type for either vector store backend (compatible with Python 3.9+)
VectorStore = Union[InMemoryVectorStore, Chroma]

# ─── Logging Configuration ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────
MAX_RESULTS = 3
CHROMA_PERSIST_DIR = "./chroma_db"

# Load environment variables
load_dotenv()


def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    """Calculate cosine similarity between two vectors.

    Cosine similarity measures the angle between two vectors, producing a
    value between -1 and 1 where 1 means identical direction.

    Args:
        vector_a: First vector.
        vector_b: Second vector (must be the same length as vector_a).

    Return:
        Cosine similarity score between -1 and 1.

    Raises:
        ValueError: If vectors have different dimensions.
    """
    if len(vector_a) != len(vector_b):
        raise ValueError("Vectors must have the same dimensions")

    a = np.array(vector_a)
    b = np.array(vector_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def load_document(
    vector_store: VectorStore,
    file_path: str,
) -> Optional[str]:
    """Load a document from a file and add it to the vector store.

    Reads the full text content of a file, creates a LangChain Document
    with metadata (fileName and createdAt), and stores it in the vector
    store for later semantic search.

    Args:
        vector_store: The vector store to add the document to.
        file_path: Path to the file to load.

    Return:
        The document ID assigned by the vector store, or None if loading
        failed.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        logger.error("File not found: %s", file_path)
        print(f"❌ Error: File not found: {file_path}")
        return None
    except Exception as e:
        logger.exception("Failed to read file: %s", file_path)
        print(f"❌ Error: Failed to read file: {file_path}")
        print(f"   {e}")
        return None

    document = Document(
        page_content=text,
        metadata={
            "fileName": os.path.basename(file_path),
            "createdAt": datetime.now().isoformat(),
        },
    )

    try:
        doc_ids = vector_store.add_documents([document])
    except Exception as e:
        error_msg = str(e)
        logger.error("Failed to add document to vector store: %s", error_msg)
        if "maximum context length" in error_msg or "token" in error_msg:
            print(f"⚠️  This document is too large to embed as a single chunk.")
            print("   Token limit exceeded. The embedding model can only process up to 8,191 tokens at once.")
            print("   Solution: The document needs to be split into smaller chunks.")
        else:
            print(f"❌ Error: {error_msg}")
        return None

    print(f"✅ Loaded '{os.path.basename(file_path)}' ({len(text):,} characters)")
    logger.info(
        "Loaded document: fileName=%s, length=%d, id=%s",
        os.path.basename(file_path),
        len(text),
        doc_ids[0] if doc_ids else "N/A",
    )

    return doc_ids[0] if doc_ids else None


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

    # Allow user to choose persistent vs ephemeral storage backend
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

    # ── Load Documents ────────────────────────────────────────────────────
    print("=== Loading Documents into Vector Database ===")

    health_doc_id = load_document(vector_store, "HealthInsuranceBrochure.md")
    if health_doc_id:
        print(f"📄 Successfully loaded HealthInsuranceBrochure.md (id: {health_doc_id})")
    else:
        print("⚠️  Could not load HealthInsuranceBrochure.md")

    employee_doc_id = load_document(vector_store, "EmployeeHandbook.md")
    if employee_doc_id:
        print(f"📄 Successfully loaded EmployeeHandbook.md (id: {employee_doc_id})")
    else:
        print("⚠️  Could not load EmployeeHandbook.md")


if __name__ == "__main__":
    main()