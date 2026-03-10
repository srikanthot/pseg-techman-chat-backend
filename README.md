# PSEG Tech Manual Chat Backend

A production-ready FastAPI backend for the PSEG field technician chat assistant.

**Hybrid RAG** (BM25 + vector) over Azure AI Search, grounded answers from Azure OpenAI,
**no API keys** — all authentication via Managed Identity (DefaultAzureCredential).

Designed to be hosted on **Azure App Service on Linux** and consumed by a
**Power Apps / PCF chat interface** via a stable HTTPS REST endpoint.

---

## Project Structure

```
pseg-techman-chat-backend/
├── app/
│   ├── __init__.py
│   ├── main.py                   # FastAPI app, CORS middleware, lifespan
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py           # All configuration from env vars (no secrets)
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py             # POST /chat, POST /chat/stream
│   │   └── schemas.py            # ChatRequest, ChatResponse, Citation
│   └── services/
│       ├── __init__.py
│       ├── credentials.py        # DefaultAzureCredential singleton
│       ├── embeddings.py         # Azure OpenAI Ada-002 embeddings (Managed Identity)
│       ├── search.py             # Hybrid Azure AI Search retrieval pipeline
│       ├── chat.py               # Azure OpenAI chat completion (streaming + non-streaming)
│       ├── citations.py          # Citation deduplication and formatting
│       └── prompts.py            # System prompt and context block formatter
├── requirements.txt
├── .env.example                  # Template — no secrets
├── .gitignore
└── README.md
```

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check — returns `{"status": "ok"}` |
| `POST` | `/chat` | **Non-streaming JSON** — primary integration endpoint for Power Apps / PCF |
| `POST` | `/chat/stream` | Streaming Server-Sent Events (SSE) — for live token display |
| `GET` | `/docs` | Swagger UI (interactive API docs) |
| `GET` | `/redoc` | ReDoc API docs |

### POST /chat — Request

```json
{
  "question": "How do I perform a pressure test on a gas service line?",
  "session_id": "optional-uuid-for-multi-turn"
}
```

### POST /chat — Response (200 OK)

```json
{
  "answer": "According to the manual [1], the pressure test procedure requires...\n\nSources:\n- gas_service_manual.pdf",
  "citations": [
    {
      "source": "gas_service_manual.pdf",
      "title": "Gas Service Installation Manual",
      "section": "Field Procedures > Pressure Testing",
      "page": "",
      "url": "https://storage.blob.core.windows.net/manuals/gas_service_manual.pdf",
      "chunk_id": "gas_service_manual.pdf_chunk_042"
    }
  ],
  "session_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

### POST /chat — HTTP Responses

| Status | Condition |
|--------|-----------|
| `200` | Success — answer and citations returned |
| `200` | Gate rejection — `answer` contains a clarifying question, `citations: []` |
| `422` | Request body validation failed (e.g. missing `question`) |
| `502` | Azure AI Search retrieval failed |
| `502` | Azure OpenAI generation failed |

### POST /chat/stream — SSE Event Types

| Event | Content |
|-------|---------|
| `data: <token>` (unnamed) | Answer token — accumulate to build the full answer |
| `event: citations` | `CitationsPayload` JSON — always emitted after the last token on success |
| `event: error` | `{"error": "..."}` JSON — emitted if retrieval or generation fails, then `[DONE]` |
| `event: ping` | Keepalive heartbeat — ignore |
| `data: [DONE]` | Stream end sentinel |

### Citation contract

**Citations are always derived from retrieval results — never from LLM answer text.**

- Citations are built immediately after retrieval succeeds and the gate passes.
- They are returned regardless of whether the LLM includes `[1]` or `Sources:` in its answer.
- This makes the citation output stable and predictable for downstream integrations.
- Inline citation markers in the answer (e.g. `[1]`) are for reader convenience only.

---

## Local Setup

### Prerequisites

- Python 3.11+
- [Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli) installed and signed in (`az login`)
- Access to the Azure OpenAI and Azure AI Search resources with the RBAC roles listed below

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/srikanthot/pseg-techman-chat-backend.git
cd pseg-techman-chat-backend

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows (Git Bash):
source .venv/Scripts/activate
# macOS / Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env — fill in AZURE_OPENAI_ENDPOINT, AZURE_SEARCH_ENDPOINT, etc.
# Do NOT add API keys — leave authentication to Managed Identity.

# 5. Log in to Azure (required for local DefaultAzureCredential)
az login

# 6. Start the development server
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000/docs to explore the API interactively.

---

## Environment Variables

All non-secret configuration. **No API keys required.**

### Azure OpenAI

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `AZURE_OPENAI_ENDPOINT` | Yes | — | Azure OpenAI resource endpoint |
| `AZURE_OPENAI_API_VERSION` | No | `2024-06-01` | API version |
| `AZURE_OPENAI_CHAT_DEPLOYMENT` | Yes | — | Chat model deployment name |
| `AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT` | Yes | — | Embeddings deployment name |
| `AZURE_OPENAI_TOKEN_SCOPE` | No | `https://cognitiveservices.azure.com/.default` | Override for GCC High: `https://cognitiveservices.azure.us/.default` |

### Azure AI Search

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `AZURE_SEARCH_ENDPOINT` | Yes | — | Azure AI Search service endpoint |
| `AZURE_SEARCH_INDEX` | No | `rag-psegtechm-index-finalv2` | Index name |

### CORS

| Variable | Default | Description |
|----------|---------|-------------|
| `ALLOWED_ORIGINS` | `http://localhost:3000,http://localhost:8000` | Comma-separated list of allowed origins. The default covers local development. **Always override in production** with your actual Power Apps / PCF domains. |

**How CORS is implemented:**

`ALLOWED_ORIGINS` is always a list of explicit origins — never a wildcard.
`allow_credentials=True` is always set, which is valid because:
- The CORS spec forbids combining `allow_origins=["*"]` with `allow_credentials=True`.
- An explicit origin list avoids this entirely — specific origins + credentials is valid.

Example for production:
```
ALLOWED_ORIGINS=https://apps.powerapps.com,https://your-org.crm.dynamics.com
```

### Index Field Mappings

| Variable | Default | Description |
|----------|---------|-------------|
| `SEARCH_CONTENT_FIELD` | `chunk` | Primary content field sent to LLM |
| `SEARCH_SEMANTIC_CONTENT_FIELD` | `chunk_for_semantic` | Field used by semantic reranker |
| `SEARCH_VECTOR_FIELD` | `text_vector` | Vector field for nearest-neighbor search |
| `SEARCH_FILENAME_FIELD` | `source_file` | Source document filename |
| `SEARCH_URL_FIELD` | `source_url` | Blob storage URL |
| `SEARCH_CHUNK_ID_FIELD` | `chunk_id` | Unique chunk identifier |
| `SEARCH_TITLE_FIELD` | `title` | Document title |
| `SEARCH_SECTION1_FIELD` | `header_1` | Top-level section heading |
| `SEARCH_SECTION2_FIELD` | `header_2` | Sub-section heading |
| `SEARCH_SECTION3_FIELD` | `header_3` | Sub-sub-section heading |
| `SEARCH_PAGE_FIELD` | *(blank)* | Page number field — leave blank if not in index |

### Retrieval Tuning

| Variable | Default | Description |
|----------|---------|-------------|
| `TOP_K` | `5` | Maximum chunks returned after all filtering |
| `RETRIEVAL_CANDIDATES` | `15` | Raw candidates fetched from Azure Search |
| `VECTOR_K` | `50` | Nearest-neighbor count for vector query |
| `USE_SEMANTIC_RERANKER` | `true` | Enable semantic reranker (requires semantic config in index) |
| `SEMANTIC_CONFIG_NAME` | `manual-semantic-config` | Semantic configuration name in the index |
| `QUERY_LANGUAGE` | `en-us` | Query language for semantic search |

### Confidence Gate

The gate evaluates the **top chunk's score** (not the average), making it practical for
technical manual Q&A where a single excellent chunk should produce a good answer.

| Variable | Default | Description |
|----------|---------|-------------|
| `MIN_RESULTS` | `1` | Minimum chunks required. Default is 1 — one relevant chunk is sufficient. Do not hard-fail just because fewer than 2 chunks were retrieved. |
| `MIN_AVG_SCORE` | `0.01` | Gate threshold for the top chunk's base RRF/hybrid score (range 0.01–0.033). |
| `MIN_RERANKER_SCORE` | `0.2` | Gate threshold for the top chunk's reranker score (range 0–4). Applied when `USE_SEMANTIC_RERANKER=true`. |

**Gate outcomes:**
- Pass → proceed to generation, return citations built from retrieval.
- Fail → return HTTP 200 with a clarifying question and `citations: []` (not a server error).

### Diversity Filtering

| Variable | Default | Description |
|----------|---------|-------------|
| `DIVERSITY_BY_SOURCE` | `true` | Enable per-source diversity capping |
| `MAX_CHUNKS_PER_SOURCE` | `2` | Standard cap: max chunks from any single source |
| `DOMINANT_SOURCE_SCORE_RATIO` | `1.5` | Score ratio to detect a dominant source |
| `MAX_CHUNKS_DOMINANT_SOURCE` | `4` | Relaxed cap for the dominant source |
| `SCORE_GAP_MIN_RATIO` | `0.55` | Discard chunks scoring below this fraction of the top score |

### Session History

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_IN_MEMORY_HISTORY` | `false` | **Disabled by default — the backend is stateless.** This is safe for App Service scale-out (instances don't share memory) and restarts (no silent data loss). Set to `true` only for single-instance local dev or demos. |

### Debug

| Variable | Default | Description |
|----------|---------|-------------|
| `TRACE_MODE` | `false` | Log detailed retrieval trace (scores, sections, content previews) |

---

## Azure App Service Deployment

### Startup Command

```
gunicorn -w 2 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 app.main:app
```

Worker count guidance: `2` for B2/B3 plans, `4` for P2v3/P3v3. Do not exceed `2 × CPU + 1`.

### App Service Configuration

1. **Runtime stack:** Python 3.11
2. **Operating system:** Linux
3. **Startup command:** (see above)
4. **App Settings:** Add all variables from `.env.example` — remember to override `ALLOWED_ORIGINS` with your production domains.
5. **System-assigned managed identity:** Enable under Identity → System assigned.

### Deployment

```bash
# ZIP deploy via Azure CLI
az webapp deploy \
  --resource-group <rg> \
  --name <app-name> \
  --src-path . \
  --type zip
```

---

## Managed Identity RBAC Requirements

Enable **system-assigned managed identity** on the App Service, then assign these roles.

### For your local development user (`az login` identity)

| Azure Resource | Role | Scope |
|----------------|------|-------|
| Azure OpenAI resource | **Cognitive Services OpenAI User** | The OpenAI resource |
| Azure AI Search resource | **Search Index Data Reader** | The Search resource |

### For the Azure App Service managed identity

| Azure Resource | Role | Scope |
|----------------|------|-------|
| Azure OpenAI resource | **Cognitive Services OpenAI User** | The OpenAI resource |
| Azure AI Search resource | **Search Index Data Reader** | The Search resource |

### Assigning roles via Azure CLI

```bash
# Get the managed identity's principal ID
PRINCIPAL_ID=$(az webapp identity show \
  --name <app-name> \
  --resource-group <rg> \
  --query principalId -o tsv)

# OpenAI role
az role assignment create \
  --assignee $PRINCIPAL_ID \
  --role "Cognitive Services OpenAI User" \
  --scope /subscriptions/<sub>/resourceGroups/<rg>/providers/Microsoft.CognitiveServices/accounts/<openai-resource>

# Search role
az role assignment create \
  --assignee $PRINCIPAL_ID \
  --role "Search Index Data Reader" \
  --scope /subscriptions/<sub>/resourceGroups/<rg>/providers/Microsoft.Search/searchServices/<search-resource>
```

> Role assignments can take up to 5 minutes to propagate.

---

## Retrieval Pipeline

```
POST /chat (or /chat/stream)
  │
  ├─ 1.  Distil keyword query (strip conversational filler → better BM25)
  ├─ 2.  Generate query embedding via Azure OpenAI Ada-002
  ├─ 3.  Hybrid search: BM25 + VectorizedQuery → RETRIEVAL_CANDIDATES results
  ├─ 4.  Optional semantic reranking (USE_SEMANTIC_RERANKER=true)
  ├─ 5.  Normalise → canonical result schema, sorted by effective score
  ├─ 6.  Filter Table-of-Contents / index pages
  ├─ 7.  Adaptive diversity cap (per-source cap, dominant-source relaxation)
  ├─ 8.  Score-gap filter (drop bottom fraction by score ratio)
  ├─ 9.  Trim to TOP_K
  ├─ 10. Confidence gate — evaluates TOP chunk score (not average):
  │        MIN_RESULTS=1: one strong chunk is sufficient
  │        Gate fail → 200 with clarifying question, citations=[]
  ├─ 11. Build citations from retrieval results
  │        Always from retrieval — never from LLM answer text
  ├─ 12. Azure OpenAI chat completion (non-streaming or streaming)
  │        Failure → HTTP 502 (JSON) or event: error (SSE)
  └─ 13. Return ChatResponse{answer, citations, session_id}
```

---

## GCC High (Azure Government) Notes

Set this variable to point to the Government cognitive services endpoint:

```
AZURE_OPENAI_TOKEN_SCOPE=https://cognitiveservices.azure.us/.default
```

All Azure SDK clients (`azure-identity`, `azure-search-documents`, `openai`) support
Azure Government endpoints when the correct endpoint URL is provided.
