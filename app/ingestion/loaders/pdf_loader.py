from pathlib import Path

import logfire
import pdfplumber
from pypdf import PdfReader

from .base import BaseLoader, LoadedDocument


class PDFLoader(BaseLoader):
    def __init__(self, min_chars_per_page: int = 20):
        self.min_chars_per_page = min_chars_per_page

    @logfire.instrument("Load PDF {file_path=}", extract_args=("file_path",))
    def load(self, file_path: str | Path) -> list[LoadedDocument]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        docs = self._load_with_pypdf(path)
        parser = "pypdf"

        if self._is_extraction_poor(docs):
            logfire.warn(
                "pypdf extraction poor, falling back to pdfplumber",
                file_name=path.name,
                doc_count=len(docs),
            )
            docs = self._load_with_pdfplumber(path)
            parser = "pdfplumber"

        if not docs:
            raise ValueError(f"No extractable text found in PDF: {path}")

        self._record_load_metrics(path, docs, file_type="pdf", parser=parser)
        return docs

    @logfire.instrument("Extract PDF with pypdf {path=}", extract_args=("path",))
    def _load_with_pypdf(self, path: Path) -> list[LoadedDocument]:
        reader = PdfReader(str(path))
        docs: list[LoadedDocument] = []

        for page_num, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue

            metadata = self._base_metadata(path, "pdf")
            metadata.update({"page": page_num, "parser": "pypdf"})
            docs.append(LoadedDocument(page_content=text, metadata=metadata))

        logfire.debug(
            "pypdf extraction complete",
            file_name=path.name,
            page_count=len(reader.pages),
            extracted_pages=len(docs),
        )
        return docs

    @logfire.instrument("Extract PDF with pdfplumber {path=}", extract_args=("path",))
    def _load_with_pdfplumber(self, path: Path) -> list[LoadedDocument]:
        docs: list[LoadedDocument] = []

        with pdfplumber.open(str(path)) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = (page.extract_text() or "").strip()
                if not text:
                    continue

                metadata = self._base_metadata(path, "pdf")
                metadata.update({"page": page_num, "parser": "pdfplumber"})
                docs.append(LoadedDocument(page_content=text, metadata=metadata))

        logfire.debug(
            "pdfplumber extraction complete",
            file_name=path.name,
            extracted_pages=len(docs),
        )
        return docs

    def _is_extraction_poor(self, docs: list[LoadedDocument]) -> bool:
        if not docs:
            return True

        avg_len = sum(len(doc.page_content) for doc in docs) / len(docs)
        return avg_len < self.min_chars_per_page
