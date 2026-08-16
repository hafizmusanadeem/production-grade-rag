from pathlib import Path

import logfire

from .base import BaseLoader, LoadedDocument


class TextLoader(BaseLoader):
    def __init__(self, encoding: str = "utf-8"):
        self.encoding = encoding

    @logfire.instrument("Load text file {file_path=}", extract_args=("file_path",))
    def load(self, file_path: str | Path) -> list[LoadedDocument]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Text file not found: {path}")

        text = path.read_text(encoding=self.encoding, errors="replace").strip()
        if not text:
            raise ValueError(f"Text file is empty: {path}")

        metadata = self._base_metadata(path, "txt")
        metadata["encoding"] = self.encoding

        docs = [LoadedDocument(page_content=text, metadata=metadata)]
        self._record_load_metrics(path, docs, file_type="txt", encoding=self.encoding)
        return docs
