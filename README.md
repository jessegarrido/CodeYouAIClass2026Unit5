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
OPENCODE_API_KEY=your-opencode-go-api-key
```

**Where to get keys:**

| Variable | Required? | Purpose | How to get |
|---|---|---|---|
| `GITHUB_TOKEN` | Recommended | Primary chat model (gpt-4o) + embeddings | [GitHub Models](https://github.com/settings/tokens) |
| `OPENCODE_API_KEY` | Recommended | Fallback chat model (GLM-5) | [OpenCode Go](https://opencode.ai/auth) (subscription required) |

> **Note:** At least one of `GITHUB_TOKEN` or `OPENCODE_API_KEY` is required for the chat model to work. `GITHUB_TOKEN` is also needed if you want document search (embeddings).

### 4. Run the App

```bash
python app.py
```

## Model Selection & Fallback Scheme

The app tries chat models in this order, testing each with a quick connectivity check to catch rate limits immediately:

```
┌─────────────────────────────────────────────┐
│  1. gpt-4o via GitHub Models (GITHUB_TOKEN) │  ← Primary (fast, high quality)
│     ↓  rate limited or unavailable          │
│  2. GLM-5 via OpenCode Go (OPENCODE_API_KEY)│  ← Fallback (good quality, low cost)
│     ↓  both unavailable                     │
│  3. ❌ App exits with error message         │
└─────────────────────────────────────────────┘
```

- **Fallback is automatic** — if gpt-4o is rate limited, the app seamlessly switches to GLM-5 without user intervention.
- **Rate limits are detected immediately** — each model is tested with a short query on startup (with `max_retries=0`), so you're not left waiting through long retry delays.

## Embeddings & Document Search

If the embeddings API (GitHub Models) is rate limited or `GITHUB_TOKEN` is not set, the app falls back to **chat-only mode**:

- The agent is created **without a search tool**
- It answers based on general knowledge instead of document content
- A clear message is shown at startup: `💬 Chat-only mode — agent created without document search tool`

This means the app is always functional even when API limits are hit — you just lose the ability to search documents.

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
