# VisualMind AI

**Semantic visual search engine for UI designers.**

Search a private image library with natural language — *"dark SaaS dashboard with sidebar nav"*, *"health app onboarding screen"*, *"warm e-commerce product cards"* — and get ranked results in under 500 ms.

---

## How It Works

1. **Upload** UI screenshots (PNG / JPEG / WebP)
2. **Auto-tag** — Google Gemini 2.5 Flash analyzes each image and extracts structured metadata: layout type, color mood, industry, UI patterns, complexity
3. **Encode** — CLIP ViT-B/32 generates a 512-dim visual embedding and stores it in Qdrant
4. **Search** — natural language queries are encoded with the same CLIP model; cosine similarity + tag keyword boost returns ranked results in real time

---

## Tech Stack

| Layer | Technology |
|---|---|
| Visual embeddings | [CLIP ViT-B/32](https://github.com/mlfoundations/open_clip) (OpenAI weights) |
| Vector database | [Qdrant](https://qdrant.tech) (cosine similarity, HNSW index) |
| VLM auto-tagging | Google Gemini 2.5 Flash |
| Backend API | FastAPI + Python 3.13 |
| Frontend | React 18 + Vite + TailwindCSS |
| Deduplication | pHash (Hamming distance < 10) |

---

## Features

- **Semantic text search** — CLIP-powered natural language to image retrieval
- **Hybrid search** — dense CLIP embeddings fused with BM25 keyword matching via Reciprocal Rank Fusion (RRF)
- **Similar image search** — find visually similar designs from any result
- **VLM auto-tagging** — Gemini extracts `layout_type`, `color_mood`, `ui_patterns`, `industry`, `complexity` for every upload
- **Tag-boosted reranking** — structured tags elevate semantically matched results above visually similar but irrelevant ones
- **Sidebar filters** — filter by layout, color mood, industry, complexity with instant re-query
- **pHash deduplication** — prevents near-duplicate images from polluting the index
- **Masonry grid** — lazy-loaded responsive image gallery with hover interactions

---

## Architecture

```
┌─────────────────┐    upload     ┌──────────────────────────────────────┐
│  React Frontend │ ────────────▶ │           FastAPI Backend            │
│  (Vite + TW)    │ ◀──results─── │                                      │
└─────────────────┘               │  ┌──────────┐   ┌─────────────────┐  │
                                  │  │   CLIP   │   │  Gemini 2.5     │  │
       text query                 │  │ ViT-B/32 │   │  Flash (tags)   │  │
       ──────────▶                │  └────┬─────┘   └────────┬────────┘  │
                                  │       │ 512-dim           │ ImageTags │
                                  │       ▼                   ▼           │
                                  │  ┌────────────────────────────────┐   │
                                  │  │         Qdrant (HNSW)          │   │
                                  │  │   vector + payload per image   │   │
                                  │  └────────────────────────────────┘   │
                                  └──────────────────────────────────────┘
```

---

## Local Setup

### Prerequisites
- Python 3.11+
- Node.js 18+
- [Qdrant binary](https://github.com/qdrant/qdrant/releases) or Docker
- Google AI Studio API key — free at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

### Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# → set GEMINI_API_KEY in .env

# Start Qdrant (without Docker)
mkdir -p /tmp/qdrant_storage && cd /tmp/qdrant_storage
/path/to/qdrant &

# Start API server
uvicorn main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/images` | List all images (gallery browse) |
| `POST` | `/api/images/upload` | Upload one or more images |
| `GET` | `/api/images/{id}` | Get single image metadata |
| `DELETE` | `/api/images/{id}` | Remove image |
| `GET` | `/api/search/text` | CLIP + tag-boosted text search |
| `GET` | `/api/search/similar/{id}` | Find visually similar images |
| `GET` | `/api/search/hybrid` | RRF fusion of dense + BM25 |
| `GET` | `/health` | Service health check |
| `GET` | `/stats` | Collection statistics |

---

## Search Quality

Text search uses two complementary signals:

1. **CLIP cosine similarity** — visual-semantic matching via `"a screenshot of a {query} user interface design"` prompt template (improves alignment vs. raw keyword encoding)
2. **Tag keyword boost** — structured Gemini tags (industry, layout, patterns) are matched against query keywords; matching tags add a calibrated score boost before final ranking

This two-stage approach ensures that `"health app"` ranks a doctor-UI landing page above a visually similar social app — even when raw CLIP scores are close.
