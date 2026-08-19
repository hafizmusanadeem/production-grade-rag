# Quick Start Guide - RAG Ingestion Pipeline

## ⚡ 5-Minute Setup

### 1. Clone & Setup (2 min)
```bash
# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
```

### 2. Get API Keys (2 min)

**Gemini API (Free)**:
1. Go to https://ai.google.dev/
2. Click "Get API Key" 
3. Copy to `.env` as `GEMINI_API_KEY`

**Qdrant (Free Cloud Tier)**:
1. Sign up at https://cloud.qdrant.io/
2. Create cluster
3. Copy URL → `QDRANT_CLUSTER_ENDPOINT`
4. Copy API key → `QDRANT_API_KEY`
5. Set `QDRANT_COLLECTION=documents`

### 3. Edit .env (1 min)
```bash
nano .env

# Must have:
GEMINI_API_KEY=your_key
QDRANT_CLUSTER_ENDPOINT=https://your-cluster.qdrant.io:6333
QDRANT_API_KEY=your_api_key
QDRANT_COLLECTION=documents
```

---

## 🚀 Common Commands

### Process Single File
```bash
python main.py process document.pdf
```

### Process All Files in Folder
```bash
python main.py process-dir ./my_documents/
```

### Process with Report
```bash
python batch_processor.py --output report.json ./documents/
```

### Resume Failed Batch
```bash
python batch_processor.py --resume report.json ./documents/
```

### Get Help
```bash
python main.py help
python batch_processor.py --help
```

---

## 📊 What Happens When You Run It

```
1️⃣  LOAD (Parse)
   document.pdf → Extract text from each page

2️⃣  CHUNK (Segment)
   50 pages → 156 chunks (1000 chars each)

3️⃣  EMBED (Vectorize)
   156 chunks → 156 vectors (768-dimensional)

4️⃣  STORE
   ├─ Local: ./DATA/embeddings/document.embeddings.json
   └─ Cloud: Qdrant collection "documents"

5️⃣  DONE ✅
   Results available for semantic search
```

---

## 🔧 Configuration

### Chunk Size Recommendation

| Use Case | Size | Overlap |
|----------|------|---------|
| Q&A / FAQ | 500 | 50 |
| General docs | 1000 | 200 |
| Long reports | 2000 | 400 |
| Code snippets | 800 | 100 |

Edit in `.env`:
```env
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

---

## ✅ Success Indicators

### Single File
```
✅ Processing: document.pdf
✅ Processing Complete!
   Chunks created: 156
   Points upserted: 156
```

### Batch Directory
```
📁 Found 5 file(s) to process
[1/5] ✅ document.pdf (156 chunks, 30.2s)
[2/5] ✅ guide.docx (89 chunks, 15.1s)
[3/5] ✅ slides.pptx (45 chunks, 12.5s)
[4/5] ✅ page.html (34 chunks, 8.3s)
[5/5] ⏭️  notes.txt (skipped: empty)

📊 Processing Summary
   ✅ Successful: 4
   ❌ Failed: 0
   📈 Total chunks: 324
```

---

## ❌ Common Issues & Fixes

### "GEMINI_API_KEY is required"
```bash
# Check 1: .env file exists?
ls -la .env

# Check 2: Key is set?
cat .env | grep GEMINI_API_KEY

# Check 3: Restart Python
# (environment variables loaded on startup)
```

### "Unsupported file type: .xyz"
```
Only supports: .pdf, .docx, .pptx, .html, .htm, .txt, .md

Convert your file or add a custom loader
```

### "No extractable text found in PDF"
```
PDF is scanned (image-based), not text-based
Options:
1. Use OCR: pip install pytesseract
2. Manually convert to PDF with text
3. Skip file
```

### Slow processing / API rate limits
```
Gemini API limit: ~100-150 chunks/minute

Solution: Process in smaller batches
python main.py process-dir ./batch1/
# Wait a bit
python main.py process-dir ./batch2/
```

### "Qdrant connection refused"
```
Check 1: Cluster running?
→ https://cloud.qdrant.io/

Check 2: Correct URL?
QDRANT_CLUSTER_ENDPOINT=https://your-cluster.qdrant.io:6333

Check 3: API key valid?
→ Regenerate in Qdrant dashboard
```

---

## 📁 File Structure After Processing

```
project/
├── .env                          # Your credentials
├── requirements.txt              # Dependencies
├── main.py                       # Single/batch processor
├── batch_processor.py            # Batch with progress
├── SETUP_GUIDE.md               # Detailed guide
├── ARCHITECTURE.md              # Technical deep dive
├── QUICK_START.md              # This file
│
└── DATA/                        # Created automatically
    └── embeddings/
        ├── document1.embeddings.json
        ├── document2.embeddings.json
        └── ...
```

---

## 📈 Performance Metrics

### Processing Speed
- **PDF**: 30+ seconds per 5MB file
- **DOCX**: 15+ seconds per 2MB file  
- **PPTX**: 25+ seconds per 10MB file
- **HTML**: 10+ seconds per 1MB file
- **TXT**: 5+ seconds per 500KB file

### Storage
- **Per chunk**: ~3-5 KB (embedding + metadata)
- **Per 500 chunks**: ~2-3 MB

### What affects speed?
1. **File complexity** (PDFs with tables are slower)
2. **Embedding API latency** (usually 0.5-1 sec per 50 chunks)
3. **Qdrant write speed** (usually <1 sec per 100 points)

---

## 🔍 Query Your Data (Next Step)

Once ingested, search your embeddings:

```python
from qdrant_client import QdrantClient
import google.generativeai as genai

# Connect
client = QdrantClient(
    url="https://your-cluster.qdrant.io:6333",
    api_key="your-api-key"
)

# Embed query
embedding = genai.embed_content(
    model="models/embedding-001",
    content="What is machine learning?"
)["embedding"]

# Search
results = client.search(
    collection_name="documents",
    query_vector=embedding,
    limit=5
)

# Results contain most similar chunks
for result in results:
    print(result.payload["page_content"])
```

---

## 📚 Next Steps

1. **Read full setup guide**: `SETUP_GUIDE.md`
2. **Understand architecture**: `ARCHITECTURE.md`
3. **Process your data**: `python main.py process-dir ./docs/`
4. **Build search interface**: Use your favorite web framework
5. **Add RAG to LLM**: Use search results as context for Claude/GPT

---

## 🆘 Getting Help

1. **Logs**: Check output messages (highly detailed)
2. **Logfire**: https://logfire.pydantic.dev/ (if enabled)
3. **Docs**: 
   - https://qdrant.tech/documentation/
   - https://ai.google.dev/tutorials
   - https://python.langchain.com/

---

## 💡 Pro Tips

✅ **Always test with 1-2 files first**
```bash
python main.py process sample.pdf
```

✅ **Use batch processor for multiple files**
```bash
python batch_processor.py --output report.json ./docs/
```

✅ **Monitor API costs**
```
Gemini embeddings: ~$0.00001 per 1000 tokens
Qdrant storage: ~$0.10 per GB/month (free tier available)
```

✅ **Back up your embeddings**
```bash
cp -r DATA/ DATA_backup/
```

✅ **Use smaller chunks for precision, larger for recall**
```
Need exact answers? → CHUNK_SIZE=500
Need broad context? → CHUNK_SIZE=2000
```

---

## 🎯 Example Workflows

### Workflow 1: Ingest Knowledge Base
```bash
# 1. Prepare documents
mkdir kb
cp ~/Documents/*.pdf kb/
cp ~/Documents/*.docx kb/

# 2. Process
python batch_processor.py --output kb_report.json kb/

# 3. Query (build search interface)
# → Use Qdrant client library
```

### Workflow 2: Add to Existing Collection
```bash
# 1. New documents
cp new_file.pdf .

# 2. Process (adds to existing collection)
python main.py process new_file.pdf

# 3. Results integrated immediately
```

### Workflow 3: Handle Large Dataset
```bash
# 1. Split into batches
ls -lR docs/ | wc -l  # Count files

# 2. Process batch by batch
python batch_processor.py docs/batch1/
python batch_processor.py docs/batch2/
python batch_processor.py docs/batch3/

# 3. All in same collection automatically
```

---

Happy processing! 🚀
