# PSEG Tech Manual Agent Backend

A production-ready FastAPI backend for the PSEG field technician chat assistant,
built on the **Microsoft Agent Framework SDK** pattern.

**Hybrid RAG** (BM25 + vector) over Azure AI Search, grounded answers from Azure OpenAI,
**no API keys** — all authentication via Managed Identity (DefaultAzureCredential).

Designed to be hosted on **Azure App Service on Linux** and consumed by a
**Power Apps / PCF chat interface** via a stable HTTPS REST endpoint.

---

## Architecture

```
POST /chat  (or /chat/stream)
     │
     ▼
routes.py                    thin: validate → AgentSession → AgentRuntime
     │
     ▼
AgentRuntime (agent_runtime/agent.py)
  │
  ├─ 1.  retrieve()          asyncio.to_thread → tools/retrieval_tool.py
  │        embed query (managed identity) → hybrid Azure AI Search (managed identity)
  │        keyword distil → VectorizedQuery → semantic reranker → diversity/gap filters
  │
  ├─ 2.  Confidence gate     score-first: top chunk score >= threshold → pass
  │        (not average-based; no hard count-only fail)
  │
  ├─ 3.  build_citations()   always from retrieval — never from LLM answer text
  │
  ├─ 4.  rag_provider        store_results(af_session, results) → session.state
  │        (RagContextProvider will read this in before_run())
  │
  └─ 5.  af_agent.run()      Microsoft Agent Framework SDK ChatAgent
              │
              ├─ InMemoryHistoryProvider.before_run()   (optional, ENABLE_IN_MEMORY_HISTORY)
              │    prepend conversation history to context
              │
              ├─ RagContextProvider.before_run()        ← key AF SDK hook
              │    pop results from session.state
              │    build_context_blocks() → numbered [N] evidence blocks
              │    context.extend_instructions() → appended to system prompt
              │
              └─ AzureOpenAIChatClient                  ← Azure OpenAI (managed identity)
                   azure_ad_token_provider=get_bearer_token_provider(
                       DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
                   streams tokens → update.text
```

---

## Microsoft Agent Framework SDK Primitives

| Primitive | Where used | Purpose |
|-----------|-----------|---------|
| `AzureOpenAIChatClient` | `llm/af_agent_factory.py` | Azure OpenAI client with managed identity |
| `client.as_agent()` | `llm/af_agent_factory.py` | Creates a ChatAgent with system prompt + providers |
| `BaseContextProvider` | `agent_runtime/af_rag_context_provider.py` | RAG context injection hook |
| `context.extend_instructions()` | `agent_runtime/af_rag_context_provider.py` | Appends retrieved chunks to system prompt |
| `InMemoryHistoryProvider` | `llm/af_agent_factory.py` | Optional multi-turn history (ENABLE_IN_MEMORY_HISTORY) |
| `AgentSession` (AF) | `agent_runtime/agent.py` | SDK session carrying history + provider state |
| `af_agent.create_session()` | `agent_runtime/agent.py` | Creates a fresh AF session per request |
| `af_agent.run(stream=True)` | `agent_runtime/agent.py` | Streams tokens as `update.text` |

**Key distinction from plain FastAPI + custom LLM loop:**
- RAG context injection is a first-class `BaseContextProvider` hook (`before_run` / `after_run`),
  not ad-hoc string concatenation inside the orchestrator.
- Conversation history is managed by `InMemoryHistoryProvider` inside the SDK session state,
  not by a hand-rolled `_histories` dict.
- The LLM call, prompt assembly, and context merging are owned by the Agent Framework SDK —
  `AgentRuntime` only needs to populate `session.state` and call `af_agent.run()`.

---

## Project Structure

```
pseg-techman-chat-backend/
├── app/
│   ├── __init__.py
│   ├── main.py                              # FastAPI app, CORS, lifespan
│   ├── config/
│   │   └── settings.py                      # All env vars — no API keys
│   ├── api/
│   │   ├── routes.py                        # POST /chat, POST /chat/stream (thin)
│   │   └── schemas.py                       # ChatRequest, ChatResponse, Citation
│   ├── agent_runtime/
│   │   ├── agent.py                         # AgentRuntime — AF SDK orchestrator
│   │   ├── session.py                       # AgentSession — per-request state
│   │   ├── af_rag_context_provider.py       # RagContextProvider(BaseContextProvider)
│   │   ├── context_providers.py             # build_context_blocks formatter
│   │   ├── citation_provider.py             # build_citations deduplicator
│   │   └── prompts.py                       # SYSTEM_PROMPT + CLARIFYING_RESPONSE
│   ├── tools/
│   │   └── retrieval_tool.py                # Hybrid Azure AI Search (managed identity)
│   └── llm/
│       ├── credentials.py                   # DefaultAzureCredential + token provider
│       ├── aoai_embeddings.py               # Query embeddings (managed identity)
│       └── af_agent_factory.py              # AzureOpenAIChatClient + as_agent()
├── requirements.txt
├── .env.example
└── README.md
```

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check — `{"status": "ok"}` |
| `POST` | `/chat` | **Non-streaming JSON** — primary endpoint for Power Apps / PCF |
| `POST` | `/chat/stream` | Streaming SSE — live token display |
| `GET` | `/docs` | Swagger UI |
| `GET` | `/redoc` | ReDoc docs |

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
  "answer": "According to the manual [1], the pressure test requires...\n\nSources:\n- gas_service_manual.pdf",
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

### POST /chat — HTTP Status Codes

| Status | Condition |
|--------|-----------|
| `200` | Success — answer and citations returned |
| `200` | Gate rejection — `answer` is a clarifying question, `citations: []` |
| `422` | Request body validation error (e.g. missing `question`) |
| `502` | Azure AI Search retrieval failed |
| `502` | Azure OpenAI generation failed |

### POST /chat/stream — SSE Event Contract

| Event | Content |
|-------|---------|
| `data: <token>` (unnamed) | Answer token — accumulate to build the full answer |
| `event: citations` | `CitationsPayload` JSON — always emitted after the last token |
| `event: error` | `{"error": "..."}` — emitted on failure, then `[DONE]` |
| `event: ping` | Keepalive heartbeat — ignore |
| `data: [DONE]` | Stream end sentinel |

### Citation contract

**Citations are always derived from retrieval results — never from LLM answer text.**

- Built from Azure AI Search results immediately after the gate passes.
- Returned regardless of whether the LLM includes `[1]` or `Sources:` in its answer.
- Stable and predictable output for downstream Power Apps / PCF integrations.

---

## Local Setup

### Prerequisites

- Python 3.11+
- [Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli) signed in (`az login`)
- Access to Azure OpenAI and Azure AI Search with the RBAC roles listed below

### Steps

```bash
# 1. Clone
git clone https://github.com/srikanthot/pseg-techman-chat-backend.git
cd pseg-techman-chat-backend

# 2. Virtual environment
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash
# source .venv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure
cp .env.example .env
# Edit .env — fill in AZURE_OPENAI_ENDPOINT, AZURE_SEARCH_ENDPOINT, deployment names.
# Do NOT add API keys.

# 5. Azure login (local DefaultAzureCredential)
az login

# 6. Start
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000/docs for interactive API docs.

---

## Environment Variables

### Azure OpenAI

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `AZURE_OPENAI_ENDPOINT` | Yes | — | Azure OpenAI resource endpoint |
| `AZURE_OPENAI_API_VERSION` | No | `2024-06-01` | API version |
| `AZURE_OPENAI_CHAT_DEPLOYMENT` | Yes | — | Chat model deployment name |
| `AZURE_OPENAI_CHAT_DEPLOYMENT_NAME` | No | same as above | Agent Framework SDK alias — keep in sync |
| `AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT` | Yes | — | Embeddings deployment name |
| `AZURE_OPENAI_TOKEN_SCOPE` | No | `https://cognitiveservices.azure.com/.default` | GCC High: `https://cognitiveservices.azure.us/.default` |

### Azure AI Search

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `AZURE_SEARCH_ENDPOINT` | Yes | — | Azure AI Search service endpoint |
| `AZURE_SEARCH_INDEX` | No | `rag-psegtechm-index-finalv2` | Index name |

### CORS

| Variable | Default | Description |
|----------|---------|-------------|
| `ALLOWED_ORIGINS` | `http://localhost:3000,http://localhost:8000` | Comma-separated allowed origins. **Always override in production.** |

### Index Field Mappings

| Variable | Default |
|----------|---------|
| `SEARCH_CONTENT_FIELD` | `chunk` |
| `SEARCH_SEMANTIC_CONTENT_FIELD` | `chunk_for_semantic` |
| `SEARCH_VECTOR_FIELD` | `text_vector` |
| `SEARCH_FILENAME_FIELD` | `source_file` |
| `SEARCH_URL_FIELD` | `source_url` |
| `SEARCH_CHUNK_ID_FIELD` | `chunk_id` |
| `SEARCH_TITLE_FIELD` | `title` |
| `SEARCH_SECTION1_FIELD` | `header_1` |
| `SEARCH_SECTION2_FIELD` | `header_2` |
| `SEARCH_SECTION3_FIELD` | `header_3` |
| `SEARCH_PAGE_FIELD` | *(blank)* |

### Retrieval Tuning

| Variable | Default | Description |
|----------|---------|-------------|
| `TOP_K` | `5` | Max chunks after all filters |
| `RETRIEVAL_CANDIDATES` | `15` | Raw pool fetched from Azure Search |
| `VECTOR_K` | `50` | kNN neighbours for vector query |
| `USE_SEMANTIC_RERANKER` | `true` | Enable semantic reranker |
| `SEMANTIC_CONFIG_NAME` | `manual-semantic-config` | Semantic config name in the index |
| `QUERY_LANGUAGE` | `en-us` | Query language for semantic search |

### Confidence Gate

Score-first evaluation — a single highly relevant chunk passes on its own.

| Variable | Default | Description |
|----------|---------|-------------|
| `MIN_RESULTS` | `1` | Secondary count guard (consulted only when top score also fails) |
| `MIN_AVG_SCORE` | `0.01` | Top chunk threshold for base RRF/hybrid score |
| `MIN_RERANKER_SCORE` | `0.2` | Top chunk threshold for semantic reranker score |

### Diversity Filtering

| Variable | Default | Description |
|----------|---------|-------------|
| `DIVERSITY_BY_SOURCE` | `true` | Enable per-source diversity cap |
| `MAX_CHUNKS_PER_SOURCE` | `2` | Standard cap per source |
| `DOMINANT_SOURCE_SCORE_RATIO` | `1.5` | Score ratio to detect dominant source |
| `MAX_CHUNKS_DOMINANT_SOURCE` | `4` | Cap for dominant source |
| `SCORE_GAP_MIN_RATIO` | `0.55` | Drop chunks below this fraction of top score |

### Session History

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_IN_MEMORY_HISTORY` | `false` | **Disabled by default (stateless).** Set `true` to enable Agent Framework `InMemoryHistoryProvider` for local dev multi-turn demos. Not suitable for production (lost on restart/scale-out). |

### Debug

| Variable | Default | Description |
|----------|---------|-------------|
| `TRACE_MODE` | `false` | Log retrieval scores, injected context blocks, diversity decisions |

---

## Azure App Service Deployment

### Startup Command

```
gunicorn -w 2 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 app.main:app
```

Worker count: `2` for B2/B3 plans, `4` for P2v3/P3v3.

### Configuration

1. **Runtime stack:** Python 3.11, Linux
2. **Startup command:** (see above)
3. **App Settings:** add all variables from `.env.example`; override `ALLOWED_ORIGINS`
4. **System-assigned managed identity:** Identity → System assigned → On

### Deploy

```bash
az webapp deploy \
  --resource-group <rg> \
  --name <app-name> \
  --src-path . \
  --type zip
```

---

## Managed Identity RBAC

Enable **system-assigned managed identity** and assign:

| Azure Resource | Role | Used by |
|----------------|------|---------|
| Azure OpenAI resource | **Cognitive Services OpenAI User** | AzureOpenAIChatClient (chat) + AzureOpenAI (embeddings) |
| Azure AI Search resource | **Search Index Data Reader** | SearchClient (retrieval) |

```bash
PRINCIPAL_ID=$(az webapp identity show \
  --name <app-name> --resource-group <rg> --query principalId -o tsv)

az role assignment create --assignee $PRINCIPAL_ID \
  --role "Cognitive Services OpenAI User" \
  --scope /subscriptions/<sub>/resourceGroups/<rg>/providers/Microsoft.CognitiveServices/accounts/<openai-resource>

az role assignment create --assignee $PRINCIPAL_ID \
  --role "Search Index Data Reader" \
  --scope /subscriptions/<sub>/resourceGroups/<rg>/providers/Microsoft.Search/searchServices/<search-resource>
```

> Role assignments take up to 5 minutes to propagate.

---

## GCC High (Azure Government)

```
AZURE_OPENAI_TOKEN_SCOPE=https://cognitiveservices.azure.us/.default
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.us/
AZURE_SEARCH_ENDPOINT=https://<search>.search.azure.us
```
