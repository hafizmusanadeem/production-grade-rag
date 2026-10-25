from pathlib import Path

import logfire
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

from .base import BaseLoader, LoadedDocument


def _shape_texts(shapes):
    texts=[]
    for shape in sorted(shapes,key=lambda s:(getattr(s,"top",0) or 0,getattr(s,"left",0) or 0)):
        if getattr(shape,"shape_type",None)==MSO_SHAPE_TYPE.GROUP:
            texts.extend(_shape_texts(shape.shapes))
        elif getattr(shape,"has_table",False) and shape.has_table:
            for row in shape.table.rows:
                cells=[c.text.strip() for c in row.cells if c.text and c.text.strip()]
                if cells: texts.append(" | ".join(cells))
        elif getattr(shape,"has_text_frame",False) and shape.has_text_frame:
            value=shape.text_frame.text.strip()
            if value: texts.append(value)
    return texts


class PPTXLoader(BaseLoader):
    @logfire.instrument("Load PPTX {file_path=}",extract_args=("file_path",))
    def load(self,file_path: str | Path)->list[LoadedDocument]:
        path=Path(file_path)
        if not path.exists(): raise FileNotFoundError(f"PPTX not found: {path}")
        presentation=Presentation(str(path)); docs=[]
        for num,slide in enumerate(presentation.slides,start=1):
            texts=_shape_texts(slide.shapes)
            title_shape=getattr(slide.shapes,"title",None)
            title=title_shape.text.strip() if title_shape is not None and getattr(title_shape,"has_text_frame",False) else ""
            if getattr(slide,"has_notes_slide",False):
                frame=getattr(slide.notes_slide,"notes_text_frame",None)
                notes=frame.text.strip() if frame is not None else ""
                if notes: texts.append("Speaker notes: "+notes)
            if not texts: continue
            metadata=self._base_metadata(path,"pptx"); metadata.update({"slide":num,"title":title})
            docs.append(LoadedDocument(page_content="\\n".join(texts),metadata=metadata))
        if not docs: raise ValueError(f"No extractable text found in PPTX: {path}")
        self._record_load_metrics(path,docs,file_type="pptx",slide_count=len(presentation.slides),extracted_slides=len(docs))
        return docs
