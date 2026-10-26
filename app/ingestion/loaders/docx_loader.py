from pathlib import Path

import logfire
from docx import Document as DocxDocument
from docx.table import Table
from docx.text.paragraph import Paragraph

from .base import BaseLoader, LoadedDocument


class DOCXLoader(BaseLoader):
    @logfire.instrument("Load DOCX {file_path=}",extract_args=("file_path",))
    def load(self,file_path: str | Path)->list[LoadedDocument]:
        path=Path(file_path)
        if not path.exists(): raise FileNotFoundError(f"DOCX not found: {path}")
        doc=DocxDocument(str(path)); parts=[]; paragraph_count=table_count=0
        for child in doc.element.body.iterchildren():
            if child.tag.endswith("}p"):
                paragraph=Paragraph(child,doc); text=paragraph.text.strip(); paragraph_count+=1
                if text:
                    style=getattr(getattr(paragraph,"style",None),"name","") or ""
                    if style.lower().startswith("heading"):
                        level="".join(c for c in style if c.isdigit())
                        text=f"{'#'*max(1,int(level or '1'))} {text}"
                    parts.append(text)
            elif child.tag.endswith("}tbl"):
                table=Table(child,doc); table_count+=1
                for row in table.rows:
                    seen=set(); cells=[]
                    for cell in row.cells:
                        key=id(cell._tc); value=cell.text.strip()
                        if key not in seen and value: seen.add(key); cells.append(value)
                    if cells: parts.append(" | ".join(cells))
        for box in doc.element.body.xpath(".//w:txbxContent"):
            values=[node.text for node in box.xpath(".//w:t") if node.text]
            text=" ".join(v.strip() for v in values if v.strip()).strip()
            if text and text not in parts: parts.append(text)
        content="\n\n".join(parts).strip()
        if not content: raise ValueError(f"No extractable text found in DOCX: {path}")
        metadata=self._base_metadata(path,"docx"); metadata.update({"paragraph_count":paragraph_count,"table_count":table_count})
        docs=[LoadedDocument(page_content=content,metadata=metadata)]
        self._record_load_metrics(path,docs,file_type="docx",paragraph_count=paragraph_count,table_count=table_count)
        return docs
