# extract_file_content.py
import os, re, json, tempfile
from typing import Optional, Tuple, Dict, Any
from pdf_extractor import extract_pdf, PDFExtractOptions

def _parse_page_range(pr: Optional[str]) -> Optional[Tuple[int,int]]:
    if not pr:
        return None
    m = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", pr)
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)))

def _json_to_plain(doc: dict) -> str:
    blocks = []
    for p in doc.get("pages", []):
        txt = p.get("text") or p.get("ocr_text") or ""
        if txt:
            blocks.append(txt.strip())
    return "\n\n".join(blocks)

def _json_to_raw(doc: dict) -> str:
    lines = []
    for p in doc.get("pages", []):
        n = p.get("page_number")
        lines.append(f"===Page {n}===")
        lines.append("{HEADER if any}")
        lines.append((p.get("text") or p.get("ocr_text") or "").strip())
        lines.append("{FOOTER if any}")
        lines.append("")
    return "\n".join(lines)

def extract_content_from_bytes(file_bytes: bytes, filename: str,
                               output_format: str = "json",
                               enable_ocr: bool = True,
                               force_ocr: bool = False,
                               lang: str = "eng",
                               tables: bool = False,
                               page_range: Optional[str] = None) -> Any:
    ext = os.path.splitext(filename)[1].lower()
    if ext != ".pdf":
        return {"error": f"Unsupported file type: {ext}"}

    # Geçici PDF yaz
    fd, temp_path = tempfile.mkstemp(suffix=ext)
    try:
        with os.fdopen(fd, "wb") as tmp:
            tmp.write(file_bytes)

        pr = _parse_page_range(page_range)

        opts = PDFExtractOptions(
            ocr_lang=lang or "eng",
            detect_tables=bool(tables),
            page_range=pr,
            raw_format=(output_format.lower() == "raw"),
            force_ocr_all=bool(force_ocr)
        )

        # enable_ocr False ise yine de motorumuz karma sayfalar için heuristik çalışır;
        # force_ocr True gelirse tüm sayfalarda OCR uygular.
        result = extract_pdf(temp_path, opts)

        fmt = output_format.lower()
        if fmt == "json":
            return result if isinstance(result, dict) else {"raw": result}
        elif fmt == "plain":
            if isinstance(result, dict):
                return _json_to_plain(result)
            return result
        elif fmt == "raw":
            return result if isinstance(result, str) else _json_to_raw(result)
        else:
            return result
    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass
