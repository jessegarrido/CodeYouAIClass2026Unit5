import logging
import os
from datetime import datetime
from typing import Optional, Union

import numpy as np
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import (
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

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


def load_document_with_chunks(
    vector_store: VectorStore,
    file_path: str,
    chunks: list[Document],
) -> int:
    """Load chunked documents into the vector store with metadata.

    Takes pre-split document chunks, enriches each with metadata including
    the source filename and chunk index, then adds them to the vector store.

    Args:
        vector_store: The vector store to add the document chunks to.
        file_path: Path to the source file (used for the fileName metadata).
        chunks: A list of LangChain Document objects, each representing a
            chunk of the original document.

    Return:
        The total number of chunks successfully stored, or 0 if all
        chunks failed to store.
    """
    total_chunks = len(chunks)
    file_name = os.path.basename(file_path)
    stored_count = 0

    print(f"📄 Loading {total_chunks} chunks from '{file_name}'...")

    for index, chunk in enumerate(chunks, 1):
        chunk.metadata["fileName"] = f"{file_name} (Chunk {index}/{total_chunks})"
        chunk.metadata["createdAt"] = datetime.now().isoformat()
        chunk.metadata["chunkIndex"] = index

        try:
            vector_store.add_documents([chunk])
            stored_count += 1
            print(f"  ✅ Chunk {index}/{total_chunks} stored")
        except Exception as e:
            logger.error("Failed to store chunk %d/%d: %s", index, total_chunks, e)
            print(f"  ❌ Chunk {index}/{total_chunks} failed: {e}")

    print(f"📁 Stored {stored_count}/{total_chunks} chunks from '{file_name}'\n")
    return stored_count


def load_with_fixed_size_chunking(
    vector_store: VectorStore,
    file_path: str,
) -> int:
    """Load a document using fixed-size character chunking.

    Reads the file, splits it into fixed-size chunks using
    CharacterTextSplitter, and stores each chunk in the vector store.

    Args:
        vector_store: The vector store to add the document chunks to.
        file_path: Path to the file to load and chunk.

    Return:
        The total number of chunks successfully stored, or 0 if the
        file could not be read.
    """
    file_name = os.path.basename(file_path)

    try:
        with open(file_path, encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        logger.error("File not found: %s", file_path)
        print(f"❌ Error: File not found: {file_path}")
        return 0
    except Exception as e:
        logger.exception("Failed to read file: %s", file_path)
        print(f"❌ Error: Failed to read file: {file_path}")
        print(f"   {e}")
        return 0

    print(f"📄 Splitting '{file_name}' ({len(text):,} characters)...")

    # Split on spaces to avoid breaking words mid-token
    splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
        separator=" ",
    )
    chunks = splitter.create_documents([text])

    # Print chunking statistics
    avg_size = sum(len(chunk.page_content) for chunk in chunks) / len(chunks) if chunks else 0
    print(f"  Created {len(chunks)} chunks (average size: {avg_size:.0f} characters)")

    stored = load_document_with_chunks(vector_store, file_path, chunks)
    return stored


def load_with_markdown_chunking(
    vector_store: VectorStore,
    file_path: str,
) -> int:
    """Load a markdown document using structure-aware chunking.

    First splits the document on markdown headers (H1 and H2) to preserve
    section structure, then applies RecursiveCharacterTextSplitter to
    further divide large sections with overlap for context continuity.

    Args:
        vector_store: The vector store to add the document chunks to.
        file_path: Path to the markdown file to load and chunk.

    Return:
        The total number of chunks successfully stored, or 0 if the
        file could not be read.
    """
    file_name = os.path.basename(file_path)

    try:
        with open(file_path, encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        logger.error("File not found: %s", file_path)
        print(f"❌ Error: File not found: {file_path}")
        return 0
    except Exception as e:
        logger.exception("Failed to read file: %s", file_path)
        print(f"❌ Error: Failed to read file: {file_path}")
        print(f"   {e}")
        return 0

    print(f"📄 Splitting '{file_name}' ({len(text):,} characters)...")

    # First pass: split on markdown headers to preserve document structure
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "Header 1"),
            ("##", "Header 2"),
        ],
    )
    md_chunks = markdown_splitter.split_text(text)
    print(f"  Markdown header split: {len(md_chunks)} sections")

    # Second pass: further split large sections with overlap for context
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_documents(md_chunks)

    # Print chunking statistics
    avg_size = sum(len(chunk.page_content) for chunk in chunks) / len(chunks) if chunks else 0
    print(f"  Created {len(chunks)} chunks (average size: {avg_size:.0f} characters)")

    stored = load_document_with_chunks(vector_store, file_path, chunks)
    return stored


def create_search_tool(vector_store: VectorStore):
    """Create a LangChain tool for searching the company document repository.

    Builds a search tool that an AI agent can use to query the vector store
    for relevant company policies, benefits, and procedures.

    Args:
        vector_store: The vector store to search against.

    Return:
        A LangChain Tool that the agent can invoke with a query string.
    """

    @tool
    def search_documents(query: str) -> str:
        """Searches the company document repository for relevant information
        based on the given query. Use this to find information about company
        policies, benefits, and procedures."""
        results = vector_store.similarity_search_with_score(query, k=3)

        if not results:
            return "No results found for the given query."

        formatted: list[str] = []
        for i, (document, score) in enumerate(results, 1):
            formatted.append(f"Result {i} (Score: {score:.4f}): {document.page_content}")

        return "\n\n".join(formatted)

    return search_documents


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

    # Create chat model using GitHub Models API
    chat_model = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        base_url="https://models.inference.ai.azure.com",
        api_key=os.getenv("GITHUB_TOKEN"),
    )
    print("🤖 Chat model: gpt-4o (temperature=0)\n")

    # ── Load Documents ────────────────────────────────────────────────────
    print("=== Loading Documents into Vector Database ===")

    health_doc_id = load_document(vector_store, "HealthInsuranceBrochure.md")
    if health_doc_id:
        print(f"📄 Successfully loaded HealthInsuranceBrochure.md (id: {health_doc_id})")
    else:
        print("⚠️  Could not load HealthInsuranceBrochure.md")

    employee_chunks = load_with_markdown_chunking(vector_store, "EmployeeHandbook.md")
    if employee_chunks > 0:
        print(f"📄 Successfully chunked and loaded EmployeeHandbook.md ({employee_chunks} chunks)")
    else:
        print("⚠️  Could not load EmployeeHandbook.md")

    # ── Create Agent ─────────────────────────────────────────────────────
    search_tool = create_search_tool(vector_store)

    agent = create_agent(
        model=chat_model,
        tools=[search_tool],
        system_prompt=(
            "You are a helpful assistant that answers questions about company "
            "policies, benefits, and procedures. Use the search_documents tool "
            "to find relevant information before answering. Always cite which "
            "document chunks you used in your answer."
        ),
    )

    print("🤖 ReAct agent created with search_documents tool\n")

    # ── Agent Chat Loop ──────────────────────────────────────────────────
    print("=" * 60)
    print("🤖 Agent Chat Interface")
    print("   Ask questions about company policies, benefits, and procedures.")
    print("   Type 'quit' or 'exit' to stop.")
    print("=" * 60)

    chat_history: list = []

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ("quit", "exit"):
            print("👋 Goodbye!")
            break

        if not user_input:
            continue

        # Build the message list for the agent
        messages = chat_history + [HumanMessage(content=user_input)]
        result = agent.invoke({"messages": messages})

        # Extract the final AI response
        ai_response = result["messages"][-1].content
        print(f"\nAgent: {ai_response}")

        # Update chat history for context continuity
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=ai_response))


if __name__ == "__main__":
    main()