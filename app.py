import logging
import os
import re
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


# ═══════════════════════════════════════════════════════════════════════════════
# Extension Challenge 1: Dynamic Chunk Sizing
# ═══════════════════════════════════════════════════════════════════════════════

def _estimate_content_density(text: str) -> float:
    """Estimate the information density of text for adaptive chunk sizing.

    Counts technical/specialized terms as a proportion of total words.
    Higher density scores indicate more dense/technical content that
    benefits from smaller chunk sizes.

    Args:
        text: The text to analyze.

    Return:
        A density score between 0.0 and 1.0.
    """
    # Keywords that suggest dense technical content
    technical_indicators = [
        "must", "shall", "required", "policy", "procedure", "compliance",
        "regulation", "coverage", "deductible", "premium", "copay",
        "enrollment", "eligibility", "benefit", "coverage", "exclusion",
        "limitation", "maximum", "minimum", "percentage", "annual",
        "effective", "termination", "provision", "agreement", "obligation",
    ]

    words = text.split()
    if not words:
        return 0.0

    lower_words = [w.lower() for w in words]
    tech_count = sum(1 for w in lower_words if w.strip(".,;:!?()[]") in technical_indicators)
    # Normalize: typical max ratio of tech terms is ~0.3, so divide by 0.3
    return min(1.0, tech_count / len(words) / 0.3)


def load_with_dynamic_chunking(
    vector_store: VectorStore,
    file_path: str,
) -> int:
    """Load a document with adaptive chunk sizing based on content density.

    Analyzes the document to estimate content density — dense technical
    sections get smaller chunks for precision, while narrative sections
    get larger chunks to preserve context flow.

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

    print(f"📄 Splitting '{file_name}' ({len(text):,} characters) with dynamic chunk sizing...")

    # First, split on markdown headers to respect document structure
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "Header 1"),
            ("##", "Header 2"),
        ],
    )
    sections = markdown_splitter.split_text(text)
    print(f"  Markdown header split: {len(sections)} sections")

    # Then apply per-section adaptive chunk sizing
    all_chunks: list[Document] = []
    for section in sections:
        density = _estimate_content_density(section.page_content)

        # Map density (0.0–1.0) to chunk size (3000–1000)
        chunk_size = max(1000, min(3000, int(3000 - density * 2000)))
        chunk_overlap = max(50, min(200, int(chunk_size * 0.1)))

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        section_chunks = splitter.split_documents([section])

        # Preserve header metadata on each sub-chunk
        for chunk in section_chunks:
            if "Header 1" in section.metadata:
                chunk.metadata["Header 1"] = section.metadata["Header 1"]
            if "Header 2" in section.metadata:
                chunk.metadata["Header 2"] = section.metadata["Header 2"]

        all_chunks.extend(section_chunks)

        # Show adaptive sizing info for the first few sections
        density_label = "high" if density > 0.4 else "medium" if density > 0.2 else "low"
        heading = section.metadata.get("Header 1", "") or section.metadata.get("Header 2", "")
        if heading:
            print(f"    [{heading}] density={density:.2f} ({density_label}) → chunk_size={chunk_size}")

    avg_size = sum(len(c.page_content) for c in all_chunks) / len(all_chunks) if all_chunks else 0
    print(f"  Created {len(all_chunks)} chunks (average size: {avg_size:.0f} characters)")

    stored = load_document_with_chunks(vector_store, file_path, all_chunks)
    return stored


# ═══════════════════════════════════════════════════════════════════════════════
# Extension Challenge 2: Hierarchical Chunking
# ═══════════════════════════════════════════════════════════════════════════════

def load_with_hierarchical_chunking(
    vector_store: VectorStore,
    file_path: str,
) -> int:
    """Load a document with two-level hierarchical chunking.

    Creates two layers of chunks:
    - **Parent chunks**: Large, context-rich sections (preserve document
      structure and narrative flow)
    - **Child chunks**: Smaller, precision-oriented pieces (enable exact
      semantic matching)

    Both layers are stored in the vector store. Child chunks carry metadata
    linking them to their parent (parentIndex), enabling cross-chunk context
    retrieval during search.

    Args:
        vector_store: The vector store to add the document chunks to.
        file_path: Path to the file to load and chunk.

    Return:
        The total number of chunks (parents + children) successfully stored.
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

    print(f"📄 Splitting '{file_name}' ({len(text):,} characters) with hierarchical chunking...")

    # ── Layer 1: Parent chunks (large, context-rich sections) ────────────
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "Header 1"),
            ("##", "Header 2"),
        ],
    )
    parent_sections = markdown_splitter.split_text(text)

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=200,
    )
    parent_chunks = parent_splitter.split_documents(parent_sections)
    print(f"  Created {len(parent_chunks)} parent chunks (context level)")

    # Store parent chunks first and record their indices
    parent_indices: list[int] = []
    for index, chunk in enumerate(parent_chunks, 1):
        chunk.metadata["fileName"] = f"{file_name} (Parent {index})"
        chunk.metadata["createdAt"] = datetime.now().isoformat()
        chunk.metadata["chunkIndex"] = index
        chunk.metadata["chunkLevel"] = "parent"
        try:
            vector_store.add_documents([chunk])
            parent_indices.append(index)
        except Exception as e:
            logger.error("Failed to store parent chunk %d: %s", index, e)

    # ── Layer 2: Child chunks (smaller, precision-oriented) ──────────────
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
    )
    # Re-split from the text directly for children
    raw_sections = markdown_splitter.split_text(text)
    raw_child_chunks = child_splitter.split_documents(raw_sections)
    print(f"  Created {len(raw_child_chunks)} child chunks (precision level)")

    # Assign each child a parent index by matching content overlap
    # Simple heuristic: assign sequential parents
    chunks_per_parent = max(1, len(raw_child_chunks) // max(1, len(parent_chunks)))

    stored_count = len(parent_indices) if parent_indices else 0
    for index, chunk in enumerate(raw_child_chunks, 1):
        parent_id = min(index, (index - 1) // chunks_per_parent + 1)
        chunk.metadata["fileName"] = f"{file_name} (Child {index}, Parent {parent_id})"
        chunk.metadata["createdAt"] = datetime.now().isoformat()
        chunk.metadata["chunkIndex"] = index
        chunk.metadata["chunkLevel"] = "child"
        chunk.metadata["parentIndex"] = parent_id
        try:
            vector_store.add_documents([chunk])
            stored_count += 1
        except Exception as e:
            logger.error("Failed to store child chunk %d: %s", index, e)

    total = len(parent_indices) + len(raw_child_chunks)
    print(f"📁 Stored {stored_count}/{total} total chunks ({len(parent_indices)} parents + {len(raw_child_chunks)} children)")
    return stored_count


# ═══════════════════════════════════════════════════════════════════════════════
# Extension Challenge 3: Chunk Quality Scoring
# ═══════════════════════════════════════════════════════════════════════════════

def score_chunk_quality(chunks: list[Document]) -> list[dict[str, object]]:
    """Score each chunk on three quality dimensions.

    Evaluates chunks for:
    - **Completeness** (0–1): Whether the chunk starts and ends at sentence
      boundaries.
    - **Coherence** (0–1): Whether the chunk appears to cover a single topic
      (estimated by vocabulary consistency).
    - **Optimal Size** (0–1): How close the chunk size is to the ideal range
      (500–1500 characters).

    Args:
        chunks: The list of Document chunks to evaluate.

    Return:
        A list of dicts, one per chunk, with keys: index, completeness,
        coherence, optimal_size, overall_score.
    """
    scores: list[dict[str, object]] = []

    for idx, chunk in enumerate(chunks):
        text = chunk.page_content.strip()
        length = len(text)

        # --- Completeness: starts/ends at sentence boundaries ---
        completeness = 1.0
        if text and not text[0].isupper():
            completeness -= 0.3
        if text and text[-1] not in (".", "!", "?", "\n"):
            completeness -= 0.3
        completeness = max(0.0, completeness)

        # --- Coherence: vocabulary overlap between first and second half ---
        mid = length // 2
        first_half = set(text[:mid].lower().split())
        second_half = set(text[mid:].lower().split())
        if first_half and second_half:
            overlap = len(first_half & second_half)
            total = len(first_half | second_half)
            coherence = overlap / total if total > 0 else 0.0
        else:
            coherence = 0.5

        # --- Optimal Size: prefer 500–1500 characters ---
        if length < 200:
            optimal_size = length / 200  # 0.0 → 1.0 as we approach 200
        elif length <= 500:
            optimal_size = 0.5 + 0.5 * (length - 200) / 300  # 0.5 → 1.0
        elif length <= 1500:
            optimal_size = 1.0  # sweet spot
        elif length <= 3000:
            optimal_size = 1.0 - 0.5 * (length - 1500) / 1500  # 1.0 → 0.5
        else:
            optimal_size = max(0.0, 0.5 - 0.5 * (length - 3000) / 2000)  # 0.5 → 0.0

        # --- Overall score (weighted average) ---
        overall = 0.3 * completeness + 0.3 * coherence + 0.4 * optimal_size

        scores.append({
            "index": idx + 1,
            "completeness": round(completeness, 3),
            "coherence": round(coherence, 3),
            "optimal_size": round(optimal_size, 3),
            "overall_score": round(overall, 3),
        })

    return scores


def load_document_with_chunks_and_scoring(
    vector_store: VectorStore,
    file_path: str,
    chunks: list[Document],
) -> int:
    """Load chunked documents with quality scoring and diagnostics.

    Extends load_document_with_chunks by also computing and displaying
    quality scores for each chunk, helping identify poorly formed chunks.

    Args:
        vector_store: The vector store to add the document chunks to.
        file_path: Path to the source file (used for the fileName metadata).
        chunks: A list of LangChain Document objects to store.

    Return:
        The total number of chunks successfully stored.
    """
    stored = load_document_with_chunks(vector_store, file_path, chunks)

    if stored > 0:
        scores = score_chunk_quality(chunks)
        # Summary statistics
        overalls = [s["overall_score"] for s in scores]
        avg_score = sum(overalls) / len(overalls) if overalls else 0.0
        min_score = min(overalls) if overalls else 0.0
        max_score = max(overalls) if overalls else 0.0

        print(f"  📊 Chunk Quality Scores:")
        print(f"     Range: {min_score:.3f} – {max_score:.3f}  |  Average: {avg_score:.3f}")
        low_quality = [s for s in scores if s["overall_score"] < 0.5]
        if low_quality:
            print(f"     ⚠️  {len(low_quality)} chunk(s) below 0.5 threshold:")
            for sq in low_quality:
                print(f"        Chunk {sq['index']}: completeness={sq['completeness']}, "
                      f"coherence={sq['coherence']}, size={sq['optimal_size']} "
                      f"→ overall={sq['overall_score']:.3f}")

    return stored


# ═══════════════════════════════════════════════════════════════════════════════
# Extension Challenge 4: Smart Overlap
# ═══════════════════════════════════════════════════════════════════════════════

def _ends_with_sentence_boundary(text: str) -> bool:
    """Check if text ends at a sentence boundary.

    Returns True if the last non-whitespace character is sentence-ending
    punctuation (. ! ?) or if the text naturally wraps (ends with \n).

    Args:
        text: The text to check.

    Return:
        True if the text ends at a sentence boundary.
    """
    stripped = text.rstrip()
    if not stripped:
        return True
    return stripped[-1] in (".", "!", "?", "\n")


def _split_with_smart_overlap(
    text: str,
    chunk_size: int,
    base_overlap: int,
) -> list[str]:
    """Split text with intelligent overlap that adds overlap only when needed.

    Examines each chunk boundary: if the split occurs mid-sentence (no
    sentence-ending punctuation), the full overlap is applied. If the
    break naturally falls at a sentence boundary, overlap is reduced or
    skipped to avoid redundancy.

    Args:
        text: The full document text to split.
        chunk_size: Target size for each chunk in characters.
        base_overlap: Maximum overlap in characters to apply when a
            mid-sentence split is detected.

    Return:
        A list of chunk text strings.
    """
    if not text:
        return []

    chunks: list[str] = []
    start = 0

    while start < len(text):
        # Determine end: aim for chunk_size from start
        end = min(start + chunk_size, len(text))

        if end >= len(text):
            # Last chunk — take everything remaining
            chunks.append(text[start:])
            break

        # Try to break at the last sentence boundary before chunk_size
        search_region = text[start:end]
        last_boundary = -1
        for punct in (". ", "!\n", "?\n", ".\n\n", ".\n"):
            pos = search_region.rfind(punct)
            if pos > last_boundary:
                last_boundary = pos

        if last_boundary > chunk_size * 0.5:
            # We found a good sentence boundary — use it
            end = start + last_boundary + 1  # +1 to include the punctuation
            chunks.append(text[start:end])

            # Check if the *next* chunk would start mid-sentence
            next_start = end
            next_end = min(next_start + chunk_size, len(text))
            if next_end < len(text):
                next_text = text[next_start:next_end]
                if not _ends_with_sentence_boundary(next_text):
                    # The break was clean — minimal or no overlap
                    start = next_start
                else:
                    # The break was clean — minimal or no overlap
                    start = next_start
            else:
                start = next_start
        else:
            # No good sentence boundary found — this is a mid-concept split
            chunks.append(text[start:end])

            # Apply full overlap since we're mid-concept
            start = end - base_overlap
            if start < 0:
                start = 0

    # Clean up: remove empty chunks
    return [c for c in chunks if c.strip()]


def load_with_smart_overlap_chunking(
    vector_store: VectorStore,
    file_path: str,
) -> int:
    """Load a document using sentence-boundary-aware smart overlap.

    Unlike fixed overlap (which blindly duplicates text at every boundary),
    smart overlap only introduces overlap when a chunk split falls in the
    middle of a sentence or concept. Natural sentence breaks get no overlap,
    reducing redundancy and storage cost while preserving context where it
    matters most.

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

    print(f"📄 Splitting '{file_name}' ({len(text):,} characters) with smart overlap...")

    raw_chunks = _split_with_smart_overlap(text, chunk_size=1500, base_overlap=200)

    # Wrap in Document objects
    chunk_docs: list[Document] = []
    for chunk_text in raw_chunks:
        chunk_docs.append(Document(page_content=chunk_text))

    # Report boundary decisions
    clean_breaks = 0
    overlapped_breaks = 0
    for chunk in chunk_docs[:-1]:  # skip last chunk
        if _ends_with_sentence_boundary(chunk.page_content):
            clean_breaks += 1
        else:
            overlapped_breaks += 1

    print(f"  Created {len(chunk_docs)} chunks")
    print(f"    Clean breaks (no overlap needed): {clean_breaks}")
    print(f"    Mid-concept breaks (overlap applied): {overlapped_breaks}")

    avg_size = sum(len(c.page_content) for c in chunk_docs) / len(chunk_docs) if chunk_docs else 0
    print(f"    Average chunk size: {avg_size:.0f} characters")

    stored = load_document_with_chunks(vector_store, file_path, chunk_docs)
    return stored


# ═══════════════════════════════════════════════════════════════════════════════
# Extension Challenge 5: Cross-Chunk Context
# ═══════════════════════════════════════════════════════════════════════════════

def create_context_aware_search_tool(vector_store: VectorStore):
    """Create a LangChain search tool that returns neighboring chunks for context.

    When a chunk is ranked highly for a query, this tool also returns the
    chunks immediately before and after it (by chunkIndex). This provides
    the agent with surrounding context, helping it answer questions where
    the answer straddles chunk boundaries.

    Args:
        vector_store: The vector store to search against.

    Return:
        A LangChain Tool that the agent can invoke with a query string.
    """

    @tool
    def search_documents_context(query: str) -> str:
        """Searches the company document repository and returns results with
        surrounding context from neighboring document chunks. Use this to
        find information about company policies, benefits, and procedures,
        especially when the answer might span multiple chunks."""
        results = vector_store.similarity_search_with_score(query, k=3)

        if not results:
            return "No results found for the given query."

        formatted: list[str] = []
        for i, (document, score) in enumerate(results, 1):
            chunk_index = document.metadata.get("chunkIndex")
            file_name = document.metadata.get("fileName", "unknown")
            chunk_level = document.metadata.get("chunkLevel", "")

            # Build the main result entry
            entry_parts = [
                f"Result {i} (Score: {score:.4f}, File: {file_name})",
                f"{document.page_content}",
            ]

            # If this chunk has a known index, try to fetch neighbors
            # by doing additional similarity searches with filters
            if chunk_index is not None and isinstance(chunk_index, int):
                neighbor_indices = [chunk_index - 1, chunk_index + 1]
                for ni in neighbor_indices:
                    if ni < 1:
                        continue
                    # InMemoryVectorStore requires a callable filter;
                    # Chroma accepts a dict, so we use a lambda for compatibility
                    neighbor_results = vector_store.similarity_search_with_score(
                        query,
                        k=5,
                        filter=lambda doc: doc.metadata.get("chunkIndex") == ni,
                    )
                    for neighbor_doc, ns in neighbor_results:
                        n_idx = neighbor_doc.metadata.get("chunkIndex")
                        if n_idx == ni:
                            relation = "previous" if ni < chunk_index else "next"
                            entry_parts.append(
                                f"  [Context - {relation} chunk (Index {ni}, Score: {ns:.4f})]: "
                                f"{neighbor_doc.page_content}"
                            )
                            break  # only include the best match for this index

            formatted.append("\n---\n".join(entry_parts))

        return "\n\n".join(formatted)

    return search_documents_context


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


def _load_with_scoring_wrapper(
    chunk_fn,
    vector_store: VectorStore,
    file_path: str,
) -> int:
    """Wrap a chunking function to use quality-scored loading (Challenge 3).

    Calls the given chunking function, then feeds its chunks through the
    quality scoring pipeline for diagnostic feedback.

    Args:
        chunk_fn: The chunking function to call.
        vector_store: The vector store to add chunks to.
        file_path: Path to the file to load and chunk.

    Return:
        The number of chunks successfully stored.
    """
    return load_document_with_chunks_and_scoring(vector_store, file_path, _get_chunks_from_fn(chunk_fn, file_path))


def _get_chunks_from_fn(chunk_fn, file_path: str) -> list[Document]:
    """Extract chunks from a chunking function without storing them.

    Replicates the chunking logic inline to extract the Document list
    that a given chunking function would create, so that quality scoring
    can be applied before storage.

    Args:
        chunk_fn: The chunking function.
        file_path: Path to the file to chunk.

    Return:
        A list of Document chunks.
    """
    # Each chunking function follows the same read-then-split pattern;
    # we replicate the chunking part to get the list of Documents.
    with open(file_path, encoding="utf-8") as f:
        text = f.read()

    if chunk_fn == load_with_markdown_chunking:
        from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
        md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "Header 1"), ("##", "Header 2")])
        md_chunks = md_splitter.split_text(text)
        splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
        return splitter.split_documents(md_chunks)

    elif chunk_fn == load_with_fixed_size_chunking:
        from langchain_text_splitters import CharacterTextSplitter
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separator=" ")
        return splitter.create_documents([text])

    elif chunk_fn == load_with_dynamic_chunking:
        # Delegate to the function but capture chunks before storage
        # by reusing its internal logic
        from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
        md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "Header 1"), ("##", "Header 2")])
        sections = md_splitter.split_text(text)

        all_chunks: list[Document] = []
        for section in sections:
            density = _estimate_content_density(section.page_content)
            chunk_size = max(1000, min(3000, int(3000 - density * 2000)))
            chunk_overlap = max(50, min(200, int(chunk_size * 0.1)))
            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            section_chunks = splitter.split_documents([section])
            for chunk in section_chunks:
                if "Header 1" in section.metadata:
                    chunk.metadata["Header 1"] = section.metadata["Header 1"]
                if "Header 2" in section.metadata:
                    chunk.metadata["Header 2"] = section.metadata["Header 2"]
            all_chunks.extend(section_chunks)
        return all_chunks

    elif chunk_fn == load_with_smart_overlap_chunking:
        raw_chunks = _split_with_smart_overlap(text, chunk_size=1500, base_overlap=200)
        return [Document(page_content=c) for c in raw_chunks]

    else:
        return []


def main() -> None:
    """Run the embedding inspector lab application."""
    logger.info("Python LangChain Agent Starting...")
    print("🤖 Python LangChain Agent Starting...\n")

    # Check for required API tokens
    github_token = os.getenv("GITHUB_TOKEN")
    opencode_api_key = os.getenv("OPENCODE_API_KEY")

    if not github_token:
        logger.error("GITHUB_TOKEN not found in environment variables")
        print("❌ Error: GITHUB_TOKEN not found in environment variables.")
        print("This is needed for generating embeddings (text-embedding-3-small).")
        print("Please create a .env file with your GitHub token:")
        print("GITHUB_TOKEN=your-github-token-here")
        print("\nGet your token from: https://github.com/settings/tokens")
        print("Or use GitHub Models: https://github.com/marketplace/models")
        return

    if not opencode_api_key:
        logger.error("OPENCODE_API_KEY not found in environment variables")
        print("❌ Error: OPENCODE_API_KEY not found in environment variables.")
        print("This is needed for the chat model (DeepSeek V4 Flash via OpenCode Go).")
        print("Please create a .env file with your OpenCode Go API key:")
        print("OPENCODE_API_KEY=your-opencode-go-api-key")
        print("\nGet your key from: https://opencode.ai/auth")
        return

    # Create embeddings instance using GitHub Models API
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        base_url="https://models.inference.ai.azure.com",
        api_key=github_token,
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

    # Create chat model using OpenCode Go API (OpenAI-compatible)
    chat_model = ChatOpenAI(
        model="deepseek-v4-flash",
        temperature=0,
        base_url="https://opencode.ai/zen/go/v1",
        api_key=opencode_api_key,
    )
    print("🤖 Chat model: DeepSeek V4 Flash (via OpenCode Go, temperature=0)\n")

    # ── Chunking Strategy Selection ───────────────────────────────────────
    print("=== Chunking Strategy Selection ===")
    print("Choose how to chunk the Employee Handbook:")
    print("  1. Markdown-aware chunking (default)")
    print("  2. Fixed-size chunking")
    print("  3. Challenge 1: Dynamic chunk sizing (adaptive chunk sizes)")
    print("  4. Challenge 2: Hierarchical chunking (parent/child chunks)")
    print("  5. Challenge 4: Smart overlap chunking (sentence-boundary aware)")
    strat_choice = input("Enter 1–5 [1]: ").strip()

    # ── Quality Scoring Toggle (Challenge 3) ──────────────────────────────
    print("\n--- Quality Scoring (Challenge 3) ---")
    score_choice = input("Enable chunk quality scoring? (y/n) [n]: ").strip().lower()
    enable_scoring = score_choice == "y"
    if enable_scoring:
        print("   📊 Quality scoring enabled — chunks scored on completeness, coherence, and size\n")

    # ── Context-Aware Search Toggle (Challenge 5) ─────────────────────────
    print("--- Context-Aware Search (Challenge 5) ---")
    context_choice = input("Enable cross-chunk context in search results? (y/n) [n]: ").strip().lower()
    enable_context_search = context_choice == "y"
    if enable_context_search:
        print("   🔗 Context-aware enabled — neighboring chunks included in search results\n")

    # ── Load Documents ────────────────────────────────────────────────────
    print("=== Loading Documents into Vector Database ===")

    health_doc_id = load_document(vector_store, "HealthInsuranceBrochure.md")
    if health_doc_id:
        print(f"📄 Successfully loaded HealthInsuranceBrochure.md (id: {health_doc_id})")
    else:
        print("⚠️  Could not load HealthInsuranceBrochure.md")

    # Map strategy choice to chunking function
    if strat_choice == "2":
        chunk_fn = load_with_fixed_size_chunking
        strategy_name = "Fixed-size chunking"
    elif strat_choice == "3":
        chunk_fn = load_with_dynamic_chunking
        strategy_name = "Dynamic chunk sizing (Challenge 1)"
    elif strat_choice == "4":
        chunk_fn = load_with_hierarchical_chunking
        strategy_name = "Hierarchical chunking (Challenge 2)"
    elif strat_choice == "5":
        chunk_fn = load_with_smart_overlap_chunking
        strategy_name = "Smart overlap chunking (Challenge 4)"
    else:
        chunk_fn = load_with_markdown_chunking
        strategy_name = "Markdown-aware chunking"

    print(f"   Strategy: {strategy_name}")

    if enable_scoring:
        employee_chunks = _load_with_scoring_wrapper(chunk_fn, vector_store, "EmployeeHandbook.md")
        print("   Quality scoring: enabled (Challenge 3)")
    else:
        employee_chunks = chunk_fn(vector_store, "EmployeeHandbook.md")

    if employee_chunks > 0:
        print(f"📄 Successfully chunked and loaded EmployeeHandbook.md ({employee_chunks} chunks)")
    else:
        print("⚠️  Could not load EmployeeHandbook.md")

    # ── Create Agent ─────────────────────────────────────────────────────
    if enable_context_search:
        search_tool = create_context_aware_search_tool(vector_store)
        print("🔗 Using context-aware search tool (Challenge 5)")
    else:
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