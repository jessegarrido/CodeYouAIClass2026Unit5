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
HF_TOKEN=hf_your_huggingface_token_here
```

**Where to get keys:**

| Variable | Required? | Purpose | How to get |
|---|---|---|---|
| `GITHUB_TOKEN` | Recommended | Chat model (gpt-4o) + Standby embeddings | [GitHub Models](https://github.com/settings/tokens) |
| `OPENCODE_API_KEY` | Optional | Fallback chat model (GLM-5) | [OpenCode Go](https://opencode.ai/auth) (subscription required) |
| `HF_TOKEN` | Recommended | Primary embeddings (HuggingFace Inference API) | [HuggingFace Settings](https://huggingface.co/settings/tokens) (free) |

> **Note:** At least one of `GITHUB_TOKEN` or `OPENCODE_API_KEY` is required for the chat model to work. `HF_TOKEN` enables primary semantic search via HuggingFace; `GITHUB_TOKEN` enables standby embeddings for runtime failover; without either, the app falls back to keyword search.

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

## Embeddings & Document Search — Three-Tier Fallback

The app cascades through three embedding providers at startup, testing each with a quick connectivity check. If one fails or is rate limited, the next is tried automatically:

```
┌──────────────────────────────────────────────────────────────┐
│  1. HuggingFace all-MiniLM-L6-v2 (HF_TOKEN)               │  ← Primary semantic search (384-dim, free)
│     ↓  rate limited / unavailable                           │
│  2. OpenAI text-embedding-3-small (GITHUB_TOKEN)            │  ← Fallback semantic search (1536-dim)
│     ↓  token missing / API error                            │
│  3. Keyword search (BM25, offline, no API needed)          │  ← Word matching, always works
└──────────────────────────────────────────────────────────────┘
```

| Tier | Method | Requires | Quality | Speed |
|---|---|---|---|---|
| **1** | HuggingFace all-MiniLM-L6-v2 (384-dim) | `HF_TOKEN` in `.env` | Good — solid semantic understanding (free) | ~100ms via Inference API |
| **2** | OpenAI text-embedding-3-small (1536-dim) | `GITHUB_TOKEN` in `.env` | Best — highest dimension, best semantic understanding | ~200ms via GitHub Models API |
| **3** | BM25 keyword scoring | Nothing — runs offline | Fair — exact word matching only | Instant (local) |

The app starts with the best available option and only falls through to the next when needed. You'll see which tier is active at startup:

- `✅ HuggingFace Inference API ready (384-dim)` — primary semantic search via free API
- `✅ OpenAI text-embedding-3-small (via GitHub Models)` — fallback semantic search
- `⚠️ No embedding provider available. Running in keyword-search mode.` — BM25 keyword fallback

### Runtime Rate-Limit Failover

If both HuggingFace and OpenAI are available, the app keeps the second provider on **standby**. When the primary embedding provider hits a rate limit (HTTP 429) during a search query, the app automatically:

1. Detects the 429 error from the embedding API
2. Rebuilds the entire vector index using the standby embeddings
3. Re-runs the search query on the new index
4. Uses the standby provider for all subsequent searches

```
Startup:  HuggingFace (primary) + OpenAI (standby)
                │
                ▼  429 rate limit during search
           ┌────────────────────────────────────┐
           │  ⚡ Automatic failover triggered     │
           │  🔄 Rebuild vector index with OpenAI │
           │  ✅ Resume search on new index        │
           └────────────────────────────────────┘
                │
                ▼
           OpenAI (now primary, no switch-back)
```

This means users won't see an error — the search seamlessly continues on the backup provider.

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
- **Runtime Failover** — Automatically switches to standby embeddings when the primary provider is rate limited
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
