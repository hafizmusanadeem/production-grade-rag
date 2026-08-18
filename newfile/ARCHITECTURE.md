# RAG Pipeline Architecture

## System Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                     RAG DATA INGESTION PIPELINE                      │
└─────────────────────────────────────────────────────────────────────┘

                         📥 INPUT DOCUMENTS
                                 │
                    ┌────────────┼────────────┐
                    │            │            │
                  .pdf         .docx        .pptx
                    │            │            │
                    └────────────┼────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   DOCUMENT ROUTER       │
                    │  (app/ingestion/loaders)│
                    └────┬────┬────┬────┬─────┘
                         │    │    │    │
        ┌────────────────┴──┬─┴─┬──┴────┼────────────┐
        │                   │   │       │            │
    PDFLoader         DOCXLoader│    HTMLLoader  TextLoader
   (pypdf +              │    PPTXLoader    (TXT/MD)
  pdfplumber)            │       (BeautifulSoup)
                         │
              ┌──────────▼──────────┐
              │ LoadedDocument      │
              │ - page_content: str │
              │ - metadata: dict    │
              └─────────┬───────────┘
                        │
        ┌───────────────▼────────────────┐
        │    CHUNKING PROCESSOR          │
        │ (app/ingestion/chunking/)      │
        │                                │
        │ RecursiveCharacterTextSplitter │
        │ Size: 1000 chars (config)      │
        │ Overlap: 200 chars (config)    │
        └───────────────┬────────────────┘
                        │
              ┌─────────▼─────────┐
              │ Chunk             │
              │ - page_content    │
              │ - metadata        │
              │ - chunk_index     │
              └─────────┬─────────┘
                        │
        ┌───────────────▼───────────────┐
        │   EMBEDDING GENERATION        │
        │ (app/services/retrieval/)     │
        │                               │
        │  Google Gemini API            │
        │  Model: gemini-embedding-001  │
        │  Output: 768-dim vectors      │
        └───────────────┬───────────────┘
                        │
              ┌─────────▼──────────┐
              │ EmbeddedChunk      │
              │ - chunk_id (UUID)  │
              │ - embedding: vec   │
              │ - page_content     │
              │ - metadata         │
              └─────────┬──────────┘
                        │
        ┌───────────────┴────────────────┐
        │                                │
        │                                │
   ┌────▼────────┐         ┌─────────────▼──────┐
   │ LOCAL STORE │         │  QDRANT VECTOR DB  │
   │             │         │                    │
   │ JSON File   │         │ Collection: docs   │
   │             │         │ Vector Index       │
   │ Format:     │         │ Cosine Distance    │
   │ {           │         │                    │
   │  "chunks":[│         │ ┌──────────────┐   │
   │    {       │         │ │ Point 1      │   │
   │      chunk_id        │ │ id: uuid     │   │
   │      embedding │         │ vector: [...] │   │
   │      content  │         │ payload: {...} │   │
   │      metadata │         │ └──────────────┘   │
   │    }         │         │ ┌──────────────┐   │
   │  ]          │         │ │ Point 2      │   │
   │ }           │         │ │ id: uuid     │   │
   │             │         │ │ vector: [...] │   │
   │ Path:       │         │ │ payload: {...} │   │
   │ ./DATA/     │         │ └──────────────┘   │
   │ embeddings/ │         │      ...          │
   └─────────────┘         └────────────────────┘

                         ✅ OUTPUT
        Embeddings stored & searchable via semantic similarity
```

---

## Component Details

### 1. Loaders (app/ingestion/loaders/)

**Purpose**: Parse various document formats

```
BaseLoader (abstract)
├── PDFLoader
│   ├── Primary: pypdf (fast, works for most PDFs)
│   └── Fallback: pdfplumber (handles scanned PDFs)
├── DOCXLoader (python-docx)
│   └── Extracts: paragraphs + tables
├── PPTXLoader (python-pptx)
│   └── Extracts: slide text + titles
├── HTMLLoader (BeautifulSoup)
│   └── Strips: scripts, styles, scripts
├── TextLoader (UTF-8 encoding)
│   └── Handles: .txt, .md files
└── Router (app/ingestion/loaders/router.py)
    └── Routes file → correct loader based on extension
```

**Output**: `LoadedDocument(page_content, metadata)`

### 2. Chunker (app/ingestion/chunking/)

**Purpose**: Split documents into overlapping chunks for embeddings

```python
RecursiveCharacterTextSplitter(
    chunk_size=1000,           # Target chunk size
    chunk_overlap=200,         # Overlap for context
    separators=[
        "\n\n",               # Paragraph breaks (best)
        "\n",                 # Line breaks
        ". ",                 # Sentences
        " ",                  # Words
        ""                    # Characters (worst)
    ]
)
```

**Why this order?** Splits intelligently while respecting semantic boundaries.

**Example**:
```
Input Document (5000 chars):
[Paragraph 1: 800 chars]
[Paragraph 2: 900 chars]
[Paragraph 3: 1200 chars]
[Paragraph 4: 1100 chars]

Output Chunks (1000 size, 200 overlap):
Chunk 1: Para1 + 200 chars of Para2 (1000 chars)
Chunk 2: Last 200 of Para1 + Para2 + 200 of Para3 (1000 chars)
Chunk 3: Last 200 of Para2 + Para3 + 200 of Para4 (1000 chars)
Chunk 4: Last 200 of Para3 + Para4 (remaining)
```

**Output**: `Chunk(page_content, metadata)`

### 3. Embeddings (app/services/retrieval/embeddings/)

**Purpose**: Convert text → semantic vectors

```
Google Gemini Embedding API
├── Model: gemini-embedding-001
├── Input: Text up to 2048 tokens
├── Output: 768-dimensional vector
├── Similarity: Cosine distance
└── Rate Limit: ~100 requests/minute
```

**What it does**: Encodes semantic meaning of text as numbers

```
"What is machine learning?" 
    → [0.234, -0.891, 0.123, ..., -0.456]  (768 dimensions)

"Tell me about AI"
    → [0.245, -0.878, 0.119, ..., -0.451]  (similar vector)

"Pizza recipes"
    → [-0.892, 0.234, -0.456, ..., 0.123]  (different vector)
```

**Cosine Similarity**: Measures angle between vectors
- Same meaning: ~0.95
- Different: ~0.10-0.30

**Output**: `EmbeddedChunk(chunk_id, embedding, page_content, metadata)`

### 4. Processor (app/ingestion/processor.py)

**Purpose**: Orchestrate entire pipeline

```python
IngestionProcessor
├── process(file_path)
│   ├── 1. load_document()       → LoadedDocument[]
│   ├── 2. chunk_documents()     → Chunk[]
│   ├── 3. embed_chunks()        → EmbeddedChunk[]
│   ├── 4. _store_locally()      → JSON file
│   └── 5. _store_in_qdrant()    → Qdrant points
└── Returns: ProcessResult
    ├── source
    ├── chunk_count
    ├── local_path
    ├── qdrant_collection
    └── qdrant_points_upserted
```

**Error Handling**:
- File not found → FileNotFoundError
- No extractable text → ValueError
- Empty file → ValueError
- Poor extraction quality → Fallback parser

### 5. Storage

#### Local Storage (JSON)
```json
{
  "source": "/full/path/to/document.pdf",
  "file_name": "document.pdf",
  "created_at": "2024-01-15T10:30:00Z",
  "embedding_model": "gemini-embedding-001",
  "chunk_count": 42,
  "chunks": [
    {
      "chunk_id": "uuid-xxxxxxxx",
      "page_content": "Text content...",
      "embedding": [0.234, -0.891, ...],  // 768 floats
      "metadata": {
        "source": "...",
        "page": 1,
        "doc_index": 0,
        "chunk_index": 0
      }
    }
  ]
}
```

**Location**: `./DATA/embeddings/{filename}.embeddings.json`

#### Qdrant Vector Database

```
Collection: documents
├── Vector Config:
│   ├── Size: 768 dimensions
│   ├── Distance: COSINE
│   └── Storage: Disk + memory index
│
├── Points:
│   ├── id: uuid (e.g., "a1b2c3d4-...")
│   ├── vector: [0.234, -0.891, ...]  (768 floats)
│   └── payload: {
│       "page_content": "Text...",
│       "source": "/path/to/file",
│       "file_name": "document.pdf",
│       "page": 1,
│       "chunk_index": 0
│     }
```

**Query**: 
```python
results = client.search(
    collection_name="documents",
    query_vector=[0.234, -0.891, ...],  # Query embedding
    limit=5
)
# Returns: Top 5 most similar chunks
```

---

## Data Flow Example

### Single PDF Processing

```
INPUT:  report.pdf (2.5 MB)
│
├─ PDFLoader (extract text from 50 pages)
│  └─ Output: 50 LoadedDocument objects
│
├─ Chunker (split into 1000-char chunks with 200 overlap)
│  └─ Output: 156 Chunk objects
│
├─ Embeddings (send to Gemini API)
│  └─ Output: 156 EmbeddedChunk objects (768-dim vectors)
│
├─ Local Storage
│  └─ ./DATA/embeddings/report.embeddings.json (saved with all data)
│
└─ Qdrant Storage
   └─ 156 Points upserted to "documents" collection

OUTPUT: ProcessResult(
  source="/path/to/report.pdf",
  chunk_count=156,
  local_path="./DATA/embeddings/report.embeddings.json",
  qdrant_collection="documents",
  qdrant_points_upserted=156
)
```

---

## Configuration Hierarchy

```
.env (Environment Variables)
  ├── GEMINI_API_KEY ──────────────┐
  ├── GEMINI_EMBEDDING_MODEL       │
  ├── CHUNK_SIZE ──────────────────┤──→ app/config.py (Settings class)
  ├── CHUNK_OVERLAP ───────────────┤──→ Used by all components
  ├── QDRANT_* ─────────────────────┤
  └── LOGFIRE_* ────────────────────┘
```

---

## Performance Characteristics

### Processing Speed
```
Document Type      File Size    Processing Time    Chunks
───────────────────────────────────────────────────────────
PDF (text)         5 MB         ~30 sec            400-500
DOCX               2 MB         ~15 sec            150-200
PPTX               10 MB        ~25 sec            200-300
HTML               1 MB         ~10 sec            80-100
TXT                500 KB       ~5 sec             40-50
```

**Bottleneck**: Gemini API embeddings (~100-150 chunks/minute)

### Storage Requirements
```
Input Document    Chunks    Local JSON    Qdrant Points
────────────────────────────────────────────────────────
5 MB PDF          400       ~8 MB         ~5 MB
2 MB DOCX         150       ~3 MB         ~2 MB
10 MB PPTX        250       ~5 MB         ~3.5 MB
```

Each embedding: 768 floats × 4 bytes = 3,072 bytes

---

## Error Handling Strategy

```
File Input
    │
    ├─ FileNotFoundError?
    │  └─ EXIT with error
    │
    ├─ Load failed?
    │  ├─ PDF: Try pypdf → pdfplumber fallback
    │  └─ Others: EXIT with error
    │
    ├─ No extractable text?
    │  └─ EXIT with error
    │
    ├─ Poor extraction quality (PDF)?
    │  └─ Switch to pdfplumber
    │
    ├─ Embedding API error?
    │  └─ Fail fast, log error, continue batch
    │
    └─ Qdrant connection error?
       └─ Still save locally, fail on upload
```

---

## Security & Privacy

### API Keys
- **GEMINI_API_KEY**: Used for embeddings generation (shared with Google)
- **QDRANT_API_KEY**: Used for vector DB authentication
- Stored in `.env` (never commit to git)
- Set via environment variables in production

### Data
- **Local embeddings**: Full embeddings + metadata saved to disk
- **Qdrant**: All content searchable (no encryption by default)
- **Logfire**: Observability logs sent to Pydantic Cloud

### Recommendations
```
✓ Use environment-specific credentials
✓ Rotate API keys regularly
✓ Use IP allowlisting on Qdrant
✓ Enable Logfire for audit trails
✓ Backup embeddings JSON locally
✓ Use private Qdrant cluster
```

---

## Scalability Considerations

### Single-threaded limitations:
- API rate limits: ~100-150 chunks/min
- Gemini embedding API: ~100 requests/min

### Bottlenecks:
1. **Embedding generation** (slowest)
   - Solution: Batch requests, parallel API calls
2. **Qdrant writes** (fast but sequential)
   - Solution: Already uses batch upsert
3. **File parsing** (fast, local)
   - Solution: Already parallelizable

### Production scaling:
```
├─ Parallel file processing (multiprocessing)
├─ Batch embedding requests (50-100 at a time)
├─ Qdrant cluster with replication
├─ Local embedding caching
└─ Async API calls
```

---

## Monitoring & Observability

### Logfire Integration
```python
logfire.instrument()          # Auto-trace functions
logfire.info("message")       # Info logs
logfire.warn("warning")       # Warnings
logfire.error("error")        # Errors
```

All processing steps are instrumented with `@logfire.instrument()` decorators.

### Metrics to monitor:
```
✓ Documents processed per day
✓ Average chunks per document
✓ Average processing time
✓ Embedding API latency
✓ Error rates by file type
✓ Qdrant collection size
✓ API costs
```

---

## Future Enhancements

```
1. Parallel processing
   └─ Process multiple documents simultaneously

2. Incremental updates
   └─ Only re-embed changed sections

3. Custom embeddings
   └─ Use open-source models (sentence-transformers)

4. Re-ranking
   └─ FlashRank cross-encoder filtering

5. Hybrid search
   └─ BM25 (lexical) + vector (semantic)

6. Metadata filtering
   └─ Search: "documents from 2024 about X"

7. Streaming ingestion
   └─ Real-time document processing

8. Multi-language support
   └─ Automatic language detection
```
