from app.observability import configure_logfire

configure_logfire()

from .base import BaseLoader, LoadedDocument
from .docx_loader import DOCXLoader
from .html_loader import HTMLLoader
from .pdf_loader import PDFLoader
from .pptx_loader import PPTXLoader
from .txt_loader import TextLoader
from .router import LOADER_MAP, load_document

__all__ = [
    "BaseLoader",
    "LoadedDocument",
    "DOCXLoader",
    "HTMLLoader",
    "PDFLoader",
    "PPTXLoader",
    "TextLoader",
    "LOADER_MAP",
    "load_document",
]
