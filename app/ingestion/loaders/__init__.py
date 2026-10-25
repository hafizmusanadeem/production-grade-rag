from .base import BaseLoader, LoadedDocument
from .docx_loader import DOCXLoader
from .html_loader import HTMLLoader
from .pdf_loader import PDFLoader
from .pptx_loader import PPTXLoader
from .router import LOADER_MAP, load_document
from .txt_loader import TextLoader

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