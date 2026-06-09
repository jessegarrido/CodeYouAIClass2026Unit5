# RAG Document Q&A Agent

A Retrieval-Augmented Generation (RAG) application that loads company documents into a vector database and provides an AI-powered chat interface for asking questions about company policies, benefits, and procedures.

## Quick Start

### 1. Create a Python Virtual Environment

```bash
# Navigate to the project directory
cd CodeYouAIClass2026Unit5

# Create a virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or if you have [uv](https://docs.astral.sh/uv/) installed:

```bash
uv pip install -r requirements.txt
```

### 3. Set Up API Keys

Create a `.env` file in the project root with the following variables:

```env
GITHUB_TOKEN=your-github-token-here
HF_TOKEN=hf_your_huggingface_token_here
OPENCODE_API_KEY=your-opencode-go-api-key
```

**Where to get keys:**

| Variable | Required? | Purpose | How to get |
|---|---|---|---|
| `GITHUB_TOKEN` | Recommended | Chat model (gpt-4o) + Fallback embeddings | [GitHub Models](https://github.com/settings/tokens) |
| `HF_TOKEN` | Recommended | Primary embeddings (HuggingFace Inference API) | [HuggingFace Settings](https://huggingface.co/settings/tokens) (free) |
| `OPENCODE_API_KEY` | Optional | Fallback chat model (GLM-5.1) | [OpenCode Go](https://opencode.ai/auth) (subscription required) |

> **Note:** At least one of `GITHUB_TOKEN` or `OPENCODE_API_KEY` is required for the chat model to work. `HF_TOKEN` enables semantic search via HuggingFace; `GITHUB_TOKEN` enables fallback semantic search via OpenAI; without either, the app falls back to keyword search.

### 4. Run the App

```bash
python app.py
```
## Features

- **AI Chat Interface** — Ask questions about company policies, benefits, and procedures
- **Document Chunking** — Multiple strategies for splitting documents into searchable pieces
  - Markdown-aware (default) — splits on headers to preserve section structure
  - Fixed-size — uniform character-based splitting
  - Dynamic — adapts chunk size based on content density
  - Hierarchical — two-level parent/child chunking
  - Smart overlap — sentence-boundary-aware overlap only where needed
- **Quality Scoring** — Optional evaluation of chunk completeness, coherence, and size
- **Context-Aware Search** — Returns neighboring chunks alongside search results
- **Persistent Storage** — Optional ChromaDB backend for vector data persistence

## Project Structure

```
├── app.py                      # Main application
├── requirements.txt            # Pinned dependencies
├── requirements.in             # Top-level dependencies (for uv)
├── .env                        # API keys (not committed to git)
├── HealthInsuranceBrochure.md  # Sample document
├── EmployeeHandbook.md         # Sample document
└── LAB*.md                     # Lab instructions
```
