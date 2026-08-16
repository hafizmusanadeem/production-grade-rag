from pathlib import Path

import logfire

from .base import BaseLoader, LoadedDocument
from .docx_loader import DOCXLoader
from .html_loader import HTMLLoader
from .pdf_loader import PDFLoader
from .pptx_loader import PPTXLoader
from .txt_loader import TextLoader

LOADER_MAP: dict[str, BaseLoader] = {
    ".pdf": PDFLoader(),
    ".html": HTMLLoader(),
    ".htm": HTMLLoader(),
    ".txt": TextLoader(),
    ".md": TextLoader(),
    ".docx": DOCXLoader(),
    ".pptx": PPTXLoader(),
}


@logfire.instrument("Route document load {file_path=}", extract_args=("file_path",))
def load_document(file_path: str | Path) -> list[LoadedDocument]:
    path = Path(file_path)
    suffix = path.suffix.lower()

    loader = LOADER_MAP.get(suffix)
    if loader is None:
        logfire.error("Unsupported file type", file_name=path.name, suffix=suffix)
        raise ValueError(f"Unsupported file type: {suffix}")

    logfire.info(
        "Routing document to loader",
        file_name=path.name,
        suffix=suffix,
        loader=loader.__class__.__name__,
    )
    return loader.load(path)
