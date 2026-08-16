from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import logfire


@dataclass
class LoadedDocument:
    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseLoader(ABC):
    @abstractmethod
    def load(self, file_path: str | Path) -> list[LoadedDocument]:
        raise NotImplementedError

    def _base_metadata(self, file_path: Path, file_type: str) -> dict[str, Any]:
        return {
            "source": str(file_path.resolve()),
            "file_name": file_path.name,
            "file_type": file_type,
        }

    def _record_load_metrics(
        self,
        path: Path,
        docs: list[LoadedDocument],
        *,
        file_type: str,
        **extra: Any,
    ) -> None:
        logfire.info(
            "Loader finished",
            loader=self.__class__.__name__,
            file_name=path.name,
            file_type=file_type,
            doc_count=len(docs),
            total_chars=sum(len(doc.page_content) for doc in docs),
            **extra,
        )
