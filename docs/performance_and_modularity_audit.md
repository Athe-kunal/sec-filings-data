# Performance and Modularity Audit

This document reviews the repository's implementation for runtime performance,
maintainability, and extensibility.

## High-impact findings

### 1) Caching is ad hoc and mostly absent in hot paths

- `company_to_ticker()` always calls `yf.Search(name)` with no in-memory or persistent cache.
- `get_cik_by_ticker()` parses SEC HTML each time for every request.
- Vector search rebuilds an embedding client repeatedly through `_make_client()`.

**Impact:** repeated network calls, higher API latency, more rate-limit risk.

**Recommendation:** introduce an explicit cache layer and cache policy:

- L1 cache (process): bounded TTL cache for `ticker->cik`, `company->ticker`, and
  repeated query embeddings.
- L2 cache (disk/redis/sqlite): optional shared cache for multi-process deployments.
- Define cache interfaces (`TickerResolver`, `CikResolver`, `EmbeddingCache`) and
  inject implementations.

### 2) Configuration boundaries are inconsistent

- Some code uses settings (`sec_settings.sec_data_dir`), while other code hardcodes
  paths (`Path("sec_data")` and `f"sec_data/{ticker}-{year}"`).
- Assertions are used for user/input validation in production flow.

**Impact:** difficult deployments, surprising behavior in non-default environments.

**Recommendation:** centralize path and request validation:

- Always derive paths from `sec_settings`.
- Replace `assert` with explicit exceptions (`ValueError`, `HTTPException`) so checks
  are never optimized away.

### 3) Service orchestration is tightly coupled

- API handlers directly call orchestration functions with mixed concerns
  (fetching/download, OCR, embedding, and response shaping).
- Similar workflows are duplicated across FastAPI and MCP.

**Impact:** harder to add new data sources and workflow variants.

**Recommendation:** use a service-layer + strategy pattern:

- `FilingsIngestionService`, `TranscriptIngestionService`, `SearchService`.
- Separate adapters for transport layers (REST, MCP, CLI).
- Add reusable command objects (`DownloadFilingCommand`, `RunOcrCommand`,
  `EmbedDocumentCommand`).

### 4) Batch concurrency is unbounded in endpoints

- Batch endpoints assemble full `asyncio.gather(*tasks)` with no explicit limiter.

**Impact:** request spikes can overwhelm SEC endpoints, Playwright, and embedding
server resources.

**Recommendation:** apply bounded concurrency at batch orchestration boundary:

- `asyncio.Semaphore` per external dependency.
- Backpressure queue for large batch requests.
- Structured retry policy with exponential backoff and jitter.

### 5) Vector metadata payload is oversized

- Upsert stores chunk text in both `documents` and `metadatas["text"]`.

**Impact:** larger storage footprint and slower writes.

**Recommendation:** store text once (in `documents`) and keep metadata minimal
(indexes, source, page, filing identifiers).

## Suggested target architecture

### A. Layered modules

- `domain/`: entities + value objects (`FilingRef`, `TranscriptRef`, `Chunk`).
- `application/`: use-cases (`ingest_filing`, `ingest_transcript`, `search`).
- `ports/`: protocol interfaces for SEC client, transcript client, OCR runner,
  vector repository, and cache.
- `adapters/`: concrete implementations (aiohttp SEC adapter, Playwright transcript
  adapter, Chroma adapter, vLLM/OpenAI embedder).
- `interfaces/`: FastAPI controllers, MCP tool handlers, CLI commands.

This keeps all transports thin and makes new workflows easier to add.

### B. Caching design

1. **Read-through cache** for lookups:
   - key examples: `company:{name}`, `ticker:{ticker}:cik`, `emb:{model}:{sha256(text)}`
2. **Content-addressable cache** for OCR and chunking:
   - cache key from source file hash + pipeline version.
3. **Invalidation strategy:**
   - TTL for external lookups.
   - versioned keys for model changes (`embedding_model`, OCR prompt version).

### C. Modularity patterns

- **Strategy pattern** for multiple transcript providers or OCR backends.
- **Factory pattern** for constructing service dependencies from settings.
- **Repository pattern** for vector and file persistence.
- **Command pattern** for idempotent ingestion steps with resumability.

## Engineering practice violations and fixes

1. **Input validation via `assert`** in runtime code.
   - Fix: explicit exceptions with user-facing error messages.
2. **Hardcoded infrastructure paths** in orchestrators.
   - Fix: route all filesystem roots through settings abstraction.
3. **Mixed abstraction levels** in long modules (especially OCR pipeline).
   - Fix: split into request builder, retry policy, page processor, and
     persistence modules with smaller functions.
4. **Cross-cutting concerns mixed in handlers** (transport + orchestration +
   formatting).
   - Fix: keep handlers as adapters that call application services.
5. **No explicit performance budgets and missing regression checks.**
   - Fix: add benchmark scripts and timing assertions in CI for hot paths.

## Prioritized implementation plan

1. Build `application/services` and `ports` interfaces; migrate FastAPI and MCP to
   call services only.
2. Add cache adapters and wire `company_to_ticker`, `get_cik_by_ticker`, and query
   embedding through cache.
3. Add bounded concurrency + retries for batch jobs and external network calls.
4. Minimize Chroma metadata footprint and add bulk-upsert instrumentation.
5. Split OCR pipeline into submodules and add unit tests around retry and parsing
   logic.

## Validation commands for this audit update

- `uv run python -m compileall .`
- `uv run python -c "import server"`
- `uv run python -c "import finance_data.server_api.models"`
