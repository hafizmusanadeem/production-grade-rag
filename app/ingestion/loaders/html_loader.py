from pathlib import Path

import logfire
from bs4 import BeautifulSoup

from .base import BaseLoader, LoadedDocument


class HTMLLoader(BaseLoader):
    @logfire.instrument("Load HTML {file_path=}", extract_args=("file_path",))
    def load(self, file_path: str | Path) -> list[LoadedDocument]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"HTML not found: {path}")

        raw_html = path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(raw_html, "html.parser")

        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        title = soup.title.get_text(strip=True) if soup.title else None
        text = soup.get_text(separator="\n", strip=True)

        if not text:
            raise ValueError(f"No extractable text found in HTML: {path}")

        metadata = self._base_metadata(path, "html")
        if title:
            metadata["title"] = title

        docs = [LoadedDocument(page_content=text, metadata=metadata)]
        self._record_load_metrics(path, docs, file_type="html", title=title)
        return docs
