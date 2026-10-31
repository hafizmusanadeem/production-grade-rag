from app.config import settings
from app.ingestion.chunking.chunker import chunk_documents
from app.ingestion.loaders.base import LoadedDocument

def _doc(text): 
    return LoadedDocument(page_content=text, metadata={"file_name": "a.docx"})

def test_no_chunk_exceeds_size_and_none_empty():
    text = ("Paragraph about Jobs. " * 40 + "\n\n") * 20
    chunks = chunk_documents([_doc(text)])
    assert chunks and all(c.page_content.strip() for c in chunks)
    assert all(len(c.page_content) <= settings.CHUNK_SIZE for c in chunks)

def test_metadata_and_indices():
    chunks = chunk_documents([_doc("x " * 3000), _doc("y " * 3000)])
    assert {c.metadata["doc_index"] for c in chunks} == {0, 1}
    d0 = [c.metadata["chunk_index"] for c in chunks if c.metadata["doc_index"] == 0]
    assert d0 == list(range(len(d0)))
    assert all(c.metadata["file_name"] == "a.docx" for c in chunks)

def test_deterministic():
    t = "Alpha beta. " * 500
    a = [c.page_content for c in chunk_documents([_doc(t)])]
    b = [c.page_content for c in chunk_documents([_doc(t)])]
    assert a == b

def test_empty_input():
    assert chunk_documents([]) == []

def test_consecutive_chunks_overlap():
    t = " ".join(f"word{i}" for i in range(1500))
    cs = [c.page_content for c in chunk_documents([_doc(t)])]
    assert cs[0].split()[-1] in cs[1]   # tail of chunk N reappears in chunk N+1
