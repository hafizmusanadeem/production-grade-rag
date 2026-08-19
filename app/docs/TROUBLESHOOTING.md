# Troubleshooting & Debugging Guide

## 🔍 Diagnostic Process

### Step 1: Check Environment
```bash
# Verify .env exists
ls -la .env

# Check all required keys are set
grep -E "GEMINI_API_KEY|QDRANT_|CHUNK_" .env

# Verify dependencies
python -c "import langchain; import qdrant_client; import google"
```

### Step 2: Test Single File
```bash
# Use a small, simple file first
python main.py process test.txt

# This tells you if the basic pipeline works
```

### Step 3: Check Logs
```bash
# Enable debug logging (add to code)
import logging
logging.basicConfig(level=logging.DEBUG)

# Or check Logfire: https://logfire.pydantic.dev/
```

---

## 📋 Error Messages & Solutions

### 1. "GEMINI_API_KEY is required"

**Cause**: Environment variable not set

**Solutions**:
```bash
# Solution 1: Check .env file
cat .env | grep GEMINI_API_KEY

# If empty, get key from:
# https://ai.google.dev/ → Get API Key

# Solution 2: Set manually in terminal
export GEMINI_API_KEY="your-key-here"

# Solution 3: Create .env file
echo "GEMINI_API_KEY=your-key-here" > .env

# Solution 4: Verify Python sees it
python -c "import os; print(os.getenv('GEMINI_API_KEY'))"
```

### 2. "QDRANT_CLUSTER_ENDPOINT is required"

**Cause**: Qdrant configuration missing

**Solutions**:
```bash
# Step 1: Create Qdrant cluster
# → https://cloud.qdrant.io/
# → Create free cluster

# Step 2: Get cluster URL
# → Shows in dashboard (e.g., https://xxx.us-east-1-0.aws.qdrant.io:6333)

# Step 3: Add to .env
echo "QDRANT_CLUSTER_ENDPOINT=https://your-cluster.qdrant.io:6333" >> .env
echo "QDRANT_API_KEY=your-api-key" >> .env
echo "QDRANT_COLLECTION=documents" >> .env

# Step 4: Verify connection
python -c "from qdrant_client import QdrantClient; c = QdrantClient(url='your-url', api_key='your-key'); print(c.get_collections())"
```

### 3. "File not found: /path/to/file.pdf"

**Cause**: Wrong file path

**Solutions**:
```bash
# Solution 1: Use absolute path
python main.py process /absolute/path/to/file.pdf

# Solution 2: Check file exists
ls -la /path/to/file.pdf

# Solution 3: Use current directory
python main.py process ./file.pdf

# Solution 4: Check working directory
pwd
```

### 4. "Unsupported file type: .xyz"

**Cause**: File extension not supported

**Solutions**:
```bash
# Supported formats:
# .pdf, .docx, .pptx, .html, .htm, .txt, .md

# Solution 1: Convert file
# PDF → text (use pdftotext or online converter)
# Word → PDF (File → Export)
# PowerPoint → PDF (File → Save As)

# Solution 2: Add custom loader
# See ARCHITECTURE.md → "Extending the Pipeline"
```

### 5. "No extractable text found in PDF"

**Cause**: PDF is scanned (image-based) or corrupted

**Solutions**:
```bash
# Solution 1: Check if text-based
pdftotext file.pdf -
# If output is empty/garbage → scanned PDF

# Solution 2: Use OCR
pip install pytesseract pillow pdf2image
# Requires Tesseract: https://github.com/UB-Mannheim/tesseract/wiki

# Solution 3: Convert PDF
# Use: Adobe Acrobat, PDF to Word service, etc.

# Solution 4: Skip file
# Remove from batch and continue
```

### 6. "Qdrant API error: Connection refused"

**Cause**: Can't connect to Qdrant cluster

**Solutions**:
```bash
# Step 1: Verify cluster is running
# → https://cloud.qdrant.io/ → Check status

# Step 2: Test URL is correct
curl -H "api-key: your-key" \
  https://your-cluster.qdrant.io:6333/health

# Step 3: Verify firewall/network
# → Check if IP is whitelisted
# → Check VPN/proxy settings

# Step 4: Recreate cluster
# → Delete old cluster
# → Create new one
# → Update .env

# Step 5: Use local Qdrant
# For development: docker run -p 6333:6333 qdrant/qdrant
# Then: QDRANT_CLUSTER_ENDPOINT=http://localhost:6333
```

### 7. "Vector size mismatch: expected 768, got 384"

**Cause**: Switched embedding model with existing collection

**Solutions**:
```bash
# Solution 1: Use new collection
# Update .env:
QDRANT_COLLECTION=documents_v2
# Creates new collection with new embedding size

# Solution 2: Recreate collection
# In Qdrant dashboard → Delete old collection
# Run processor again (creates new one)

# Solution 3: Use same embedding model
# Check which model your collection expects:
python -c "
from qdrant_client import QdrantClient
c = QdrantClient(url='...', api_key='...')
info = c.get_collection('documents')
print(info.config.params.vectors.size)
"
# Use embedding model that produces this size
```

### 8. "API rate limit exceeded"

**Cause**: Too many requests to Gemini API too quickly

**Solutions**:
```bash
# Solution 1: Wait and retry
# Gemini limit: ~100 requests/minute
# Batch is 5 files? → Wait 3 minutes between runs

# Solution 2: Reduce batch size
# Process 1-2 files at a time
python main.py process file1.pdf
# Wait
python main.py process file2.pdf

# Solution 3: Implement backoff
# Add to app/services/retrieval/embeddings/embeddings.py:
import time

def embed_chunks(chunks):
    # ...
    for chunk in chunks:
        result = client.embed_documents([chunk.page_content])
        time.sleep(0.5)  # Wait 500ms between requests
```

### 9. "Empty response from embedding API"

**Cause**: API error or network issue

**Solutions**:
```bash
# Step 1: Check API key is valid
# Try in Python:
import google.generativeai as genai
genai.configure(api_key="your-key")
result = genai.embed_content(
    model="models/embedding-001",
    content="test"
)
print(result)

# Step 2: Check content length
# Gemini limit: 2048 tokens (~8000 chars)
text = "..."
if len(text) > 8000:
    print("Text too long for API")

# Step 3: Check network
ping google.com

# Step 4: Try alternative model
# Update .env:
GEMINI_EMBEDDING_MODEL=text-embedding-004
```

### 10. "Logfire token invalid"

**Cause**: Logfire token wrong or revoked

**Solutions**:
```bash
# Solution 1: Regenerate token
# https://logfire.pydantic.dev/ → Settings → Regenerate

# Solution 2: Disable Logfire (optional)
# Set in .env:
LOGFIRE_TOKEN=

# Solution 3: Check token format
# Should be a long alphanumeric string
echo $LOGFIRE_TOKEN | wc -c
# Should be > 30 characters
```

---

## 🐛 Silent Failures (No Error, But Wrong Result)

### Issue: File processed but no chunks created

```bash
# Debug steps:
1. Check if text was extracted
   python -c "
   from app.ingestion.loaders.pdf_loader import PDFLoader
   loader = PDFLoader()
   docs = loader.load('file.pdf')
   print(f'Extracted: {len(docs)} docs')
   for doc in docs:
       print(f'Length: {len(doc.page_content)}')
   "

2. Check chunking
   python -c "
   from app.ingestion.chunking import chunk_documents
   from app.ingestion.loaders import load_document
   docs = load_document('file.pdf')
   chunks = chunk_documents(docs)
   print(f'Chunks: {len(chunks)}')
   "

3. Check embeddings
   python -c "
   from app.services.retrieval.embeddings import embed_chunks
   # ... use chunks from above
   embedded = embed_chunks(chunks)
   print(f'Embedded: {len(embedded)}')
   "
```

### Issue: Chunks created but not in Qdrant

```bash
# Check 1: Local storage
ls -la DATA/embeddings/
# Should have *.embeddings.json file

# Check 2: Verify Qdrant connection
python -c "
from qdrant_client import QdrantClient
c = QdrantClient(url='...', api_key='...')
collections = c.get_collections()
print('Collections:', [c.name for c in collections.collections])
"

# Check 3: Count points in collection
python -c "
from qdrant_client import QdrantClient
c = QdrantClient(url='...', api_key='...')
info = c.get_collection('documents')
print(f'Points: {info.points_count}')
"

# Check 4: Verify upload succeeded
# Look at JSON file:
cat DATA/embeddings/file.embeddings.json | python -m json.tool
# Should show all chunks
```

### Issue: Processed multiple times (duplicates in Qdrant)

```bash
# Problem: Same file processed multiple times, chunks duplicated

# Solution 1: Clear collection
from qdrant_client import QdrantClient
client = QdrantClient(url='...', api_key='...')
client.delete_collection('documents')
# Then reprocess

# Solution 2: Use new collection
# .env: QDRANT_COLLECTION=documents_v2

# Solution 3: Keep track of processed files
# Use batch_processor.py --output report.json
# Then --resume report.json (skips already processed)
```

---

## 🔧 Advanced Debugging

### Enable Verbose Logging

```python
# Add to main.py before running:
import logging

# Set root logger to DEBUG
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Or specific modules
logging.getLogger("app").setLevel(logging.DEBUG)
logging.getLogger("qdrant").setLevel(logging.DEBUG)
```

### Instrument Custom Code

```python
import logfire

@logfire.instrument("My operation")
def my_function():
    logfire.info("Step 1")
    # ... code ...
    logfire.info("Step 2")
    logfire.debug("Detailed info", extra_data="value")
```

### Profile Performance

```python
import time

# Timing wrapper
def time_it(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time() - start:.2f}s")
        return result
    return wrapper

@time_it
def load_document(path):
    # ...
    pass
```

### Test API Connectivity

```python
# Test Gemini API
import google.generativeai as genai
genai.configure(api_key="your-key")
result = genai.embed_content(
    model="models/embedding-001",
    content="test"
)
print(f"Success: {len(result['embedding'])} dimensions")

# Test Qdrant API
from qdrant_client import QdrantClient
client = QdrantClient(
    url="https://your-cluster.qdrant.io:6333",
    api_key="your-key"
)
print(client.get_collections())
```

---

## 📊 System Health Check

Run this script to diagnose everything:

```python
#!/usr/bin/env python3
import sys
import os

print("🔍 SYSTEM HEALTH CHECK\n")

# 1. Python version
import sys
print(f"✓ Python {sys.version.split()[0]}")

# 2. Environment variables
vars = ["GEMINI_API_KEY", "QDRANT_API_KEY", "QDRANT_CLUSTER_ENDPOINT"]
for v in vars:
    status = "✓" if os.getenv(v) else "✗"
    print(f"{status} {v}: {'set' if os.getenv(v) else 'MISSING'}")

# 3. Dependencies
deps = [
    "langchain",
    "qdrant_client", 
    "google.generativeai",
    "pydantic",
    "logfire"
]
for dep in deps:
    try:
        __import__(dep.replace('.', '_'))
        print(f"✓ {dep}")
    except ImportError:
        print(f"✗ {dep}: MISSING")

# 4. API Connectivity
print("\n🔌 API Tests:")

# Gemini
try:
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    result = genai.embed_content(
        model="models/embedding-001",
        content="test"
    )
    print(f"✓ Gemini API: {len(result['embedding'])} dims")
except Exception as e:
    print(f"✗ Gemini API: {e}")

# Qdrant
try:
    from qdrant_client import QdrantClient
    client = QdrantClient(
        url=os.getenv("QDRANT_CLUSTER_ENDPOINT"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    collections = client.get_collections()
    print(f"✓ Qdrant API: {len(collections.collections)} collections")
except Exception as e:
    print(f"✗ Qdrant API: {e}")

print("\n✅ Health check complete!")
```

---

## 🚀 Performance Troubleshooting

### Processing is slow

```bash
# Check 1: Is it the file parsing?
time python -c "
from app.ingestion.loaders import load_document
docs = load_document('file.pdf')
print(f'Loaded: {len(docs)} docs')
"

# Check 2: Is it the chunking?
time python -c "
from app.ingestion.loaders import load_document
from app.ingestion.chunking import chunk_documents
docs = load_document('file.pdf')
chunks = chunk_documents(docs)
print(f'Chunks: {len(chunks)}')
"

# Check 3: Is it the embedding?
# (Usually this is slowest - API calls)
time python -c "
from app.services.retrieval.embeddings import embed_chunks
# ... chunks from above
embedded = embed_chunks(chunks)
print(f'Embedded: {len(embedded)}')
"
```

### Qdrant writes are slow

```bash
# Check cluster status
curl -H "api-key: your-key" \
  https://your-cluster.qdrant.io:6333/health

# Check if collection is too large
python -c "
from qdrant_client import QdrantClient
c = QdrantClient(url='...', api_key='...')
info = c.get_collection('documents')
print(f'Points: {info.points_count}')
print(f'Size: {info.data_disk_size} bytes')
"

# Consider splitting into multiple collections
# if > 1 million points
```

---

## 📞 Getting Support

If you're stuck:

1. **Run health check** (above)
2. **Check logs** at https://logfire.pydantic.dev/
3. **Read error message carefully** - usually tells you what's wrong
4. **Search GitHub issues** - you're likely not first
5. **Check API status pages**:
   - https://status.cloud.google.com/ (Gemini)
   - https://status.qdrant.io/ (Qdrant)

---

## 🎯 Success Checklist

- [ ] `.env` file exists with all 4 required variables
- [ ] API keys are valid and active
- [ ] At least 1 small test file processes successfully
- [ ] Local embeddings file created in `DATA/embeddings/`
- [ ] Qdrant collection created and contains points
- [ ] Batch processing works on 3+ files
- [ ] Logfire shows processing steps (if enabled)

If all ✓, your pipeline is ready to ingest large datasets! 🚀
