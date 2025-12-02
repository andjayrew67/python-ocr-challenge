# pdf_extractor.py
# Robust, page-wise PDF extractor with OCR fallback, tables, forms, links, images, footer artifacts.
# Python 3.10+

from __future__ import annotations
import os, io, json, re, shutil, tempfile, subprocess
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
import concurrent.futures

# --- Third-party core deps (install via pip): PyMuPDF, Pillow, pytesseract
import fitz  # PyMuPDF
from PIL import Image, ImageOps
import pytesseract

# --- Optional deps (used if available): camelot, pdfplumber, pandas
try:
    import camelot  # type: ignore
    _HAS_CAMELOT = True
except Exception:
    _HAS_CAMELOT = False

try:
    import pdfplumber  # type: ignore
    _HAS_PDFPLUMBER = True
except Exception:
    _HAS_PDFPLUMBER = False


# =========================
# Options & Data Structures
# =========================

@dataclass
class PDFExtractOptions:
    dpi: int = 200
    ocr_lang: str = "eng"
    max_workers: Optional[int] = None   # None -> auto
    use_ocrmypdf: bool = True
    detect_tables: bool = True
    table_engine_priority: Tuple[str, ...] = ("camelot", "pdfplumber")  # try in order if installed
    include_images_metadata: bool = True
    save_footer_snapshot: bool = False
    footer_snapshot_size: Tuple[int, int] = (120, 120)  # w, h in px (on rasterized page)
    footer_threshold_ratio: float = 0.10  # bottom 10% is "footer" area
    output_image_dir: Optional[str] = None  # if set, save image exports here
    keep_intermediate: bool = False  # keep ocrmypdf outputs
    page_range: Optional[Tuple[int, int]] = None  # (start, end) 1-based inclusive
    raw_format: bool = False  # return plain text with ===Page N=== blocks (instead of JSON)
    add_footer_artifacts: bool = True

    # --- Advanced / Best mode toggles ---
    force_ocr_all: bool = False              # Force OCR on all pages
    min_chars_without_footer: int = 50       # If non-footer text chars < threshold -> OCR candidate
    min_large_image_area_ratio: float = 0.20 # If largest raster image area/page area >= threshold -> OCR candidate


@dataclass
class PageResult:
    page_number: int
    text: str = ""
    ocr_text: str = ""
    tables: List[Dict[str, Any]] = field(default_factory=list)
    images: List[Dict[str, Any]] = field(default_factory=list)
    links: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


# ===========
# Small Utils
# ===========

def _which(cmd: str) -> bool:
    return shutil.which(cmd) is not None

def _safe_int(n) -> int:
    try:
        return int(n)
    except Exception:
        return 0

def _iter_pages(doc: fitz.Document, page_range: Optional[Tuple[int,int]]) -> List[int]:
    total = doc.page_count
    if not page_range:
        return list(range(1, total+1))
    start, end = page_range
    start = max(1, start)
    end = min(total, end)
    if start > end:
        return list(range(1, total+1))
    return list(range(start, end+1))

def _extract_links(page: fitz.Page) -> List[Dict[str, Any]]:
    out = []
    try:
        for lnk in page.get_links():
            target = lnk.get("uri") or lnk.get("file") or lnk.get("dest")
            if target:
                out.append({"target": str(target)})
    except Exception as e:
        out.append({"error": f"link_extraction_failed: {e}"})
    return out

def _extract_images_metadata(doc: fitz.Document, page: fitz.Page) -> List[Dict[str, Any]]:
    metas = []
    try:
        for info in page.get_images(full=True):
            xref = info[0]
            pm = fitz.Pixmap(doc, xref)
            meta = {"xref": xref, "width": pm.width, "height": pm.height, "alpha": pm.alpha, "n": pm.n}
            metas.append(meta)
    except Exception as e:
        metas.append({"error": f"image_meta_failed: {e}"})
    return metas

def _footer_rect(page: fitz.Page, ratio: float) -> fitz.Rect:
    r = page.rect
    h = r.height
    y0 = r.y1 - h*ratio
    return fitz.Rect(r.x0, y0, r.x1, r.y1)

def _footer_link(links: List[Dict[str, Any]], page: fitz.Page, ratio: float) -> Optional[str]:
    try:
        bottom_y = page.rect.y1 - page.rect.height * ratio
        for lnk in page.get_links():
            tgt = lnk.get("uri") or lnk.get("file") or lnk.get("dest")
            rect = lnk.get("from")
            if tgt and rect and rect[1] >= bottom_y:
                return str(tgt)
    except Exception:
        pass
    for l in reversed(links):
        if "target" in l:
            return l["target"]
    return None

def _rasterize_footer_snapshot(page_img: Image.Image, size: Tuple[int,int]) -> Image.Image:
    W, H = page_img.size
    w, h = size
    left, top = 0, max(0, H - h)
    right, bottom = min(W, w), H
    crop = page_img.crop((left, top, right, bottom))
    return crop

def _page_to_image(page: fitz.Page, dpi: int) -> Image.Image:
    pix = page.get_pixmap(dpi=dpi)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return ImageOps.exif_transpose(img)

def _ocr_image(img: Image.Image, lang: str) -> str:
    try:
        return pytesseract.image_to_string(img, lang=lang) or ""
    except Exception as e:
        return f"[OCR_ERROR] {e}"

def _has_meaningful_text(txt: str) -> bool:
    return bool(txt and txt.strip() and re.search(r"\w", txt))

def _extract_form_fields(doc: fitz.Document) -> Dict[str, Any]:
    fields = {}
    try:
        for page_index in range(doc.page_count):
            p = doc.load_page(page_index)
            if not hasattr(p, "widgets") or not p.widgets:
                continue
            for w in p.widgets():
                try:
                    nm = w.field_name
                    val = w.field_value
                    if nm:
                        fields[str(nm)] = str(val) if val is not None else ""
                except Exception:
                    continue
    except Exception:
        pass
    return fields


# =========================
# Text Post-Processing (NEW)
# =========================

_RE_MANY_BLANKS = re.compile(r"\n{3,}")
_RE_FOOTER_PAGE = re.compile(r"^\s*page\s+\d+\s+of\s+\d+\s*$", re.I)
_RE_URL_LINE     = re.compile(r"^\s*https?://\S+\s*$", re.I)
_RE_NOISE_LINES  = re.compile(r"^\s*(J|><o|░)\s*$")

def _normalize_quotes(s: str) -> str:
    return (s.replace("“", '"').replace("”", '"')
             .replace("‘", "'").replace("’", "'")
             .replace("´", "'").replace("`", "'")
             .replace("˝", '"').replace("„", '"')
             .replace("Æ", "'"))

def _fix_common_ocr(s: str) -> str:
    s = s.replace("IÆm", "I'm").replace("donÆt", "don't").replace("IÆll", "I'll")
    s = s.replace("IÆve", "I've").replace("canÆt", "can't").replace("isnÆt", "isn't")
    s = re.sub(r"\b\|\b", "I", s)
    return s

def _strip_footer_noise(lines: List[str]) -> List[str]:
    out = []
    for ln in lines:
        if _RE_NOISE_LINES.match(ln):
            continue
        out.append(ln)
    while out and (_RE_URL_LINE.match(out[-1]) or _RE_FOOTER_PAGE.match(out[-1])):
        out.pop()
    return out

def _dedup_lines(lines: List[str]) -> List[str]:
    out = []
    prev = None
    for ln in lines:
        if ln.strip() and ln.strip() == (prev.strip() if prev else None):
            continue
        out.append(ln)
        prev = ln
    return out

def _postprocess_text(s: str) -> str:
    if not s:
        return ""
    s = _normalize_quotes(s)
    s = _fix_common_ocr(s)
    raw_lines = s.splitlines()
    lines = [ln.rstrip() for ln in raw_lines]
    lines = _strip_footer_noise(lines)
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    lines = _dedup_lines(lines)
    s = "\n".join(lines)
    s = _RE_MANY_BLANKS.sub("\n\n", s)
    return s.strip()


# ===========================
# Native (non-OCR) Extraction
# ===========================

def _tables_via_camelot(pdf_path: str, pages_spec: str) -> Dict[int, List[Dict[str, Any]]]:
    out: Dict[int, List[Dict[str, Any]]] = {}
    for flavor in ("lattice", "stream"):
        try:
            tables = camelot.read_pdf(pdf_path, pages=pages_spec, flavor=flavor)
            for i, t in enumerate(tables, start=1):
                pg = _safe_int(t.page)
                data = t.df.fillna("").values.tolist()
                out.setdefault(pg, []).append({
                    "table_number": len(out.get(pg, [])) + 1,
                    "data": data,
                    "engine": f"camelot-{flavor}"
                })
        except Exception:
            continue
    return out

def _tables_via_pdfplumber(pdf_path: str, page_nums: List[int]) -> Dict[int, List[Dict[str, Any]]]:
    out: Dict[int, List[Dict[str, Any]]] = {}
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for pg in page_nums:
                idx = pg - 1
                if idx < 0 or idx >= len(pdf.pages):
                    continue
                page = pdf.pages[idx]
                try:
                    ts = page.extract_tables() or []
                    for _t_i, table in enumerate(ts, start=1):
                        data = [[c if c is not None else "" for c in row] for row in table]
                        out.setdefault(pg, []).append({
                            "table_number": len(out.get(pg, [])) + 1,
                            "data": data,
                            "engine": "pdfplumber"
                        })
                except Exception:
                    continue
    except Exception:
        pass
    return out

def _merge_tables(pref: Dict[int, List[Dict[str, Any]]], fallback: Dict[int, List[Dict[str, Any]]]) -> Dict[int, List[Dict[str, Any]]]:
    result = {k: v[:] for k, v in pref.items()}
    for pg, tbls in fallback.items():
        result.setdefault(pg, [])
        result[pg].extend(tbls)
    return result

def _non_footer_text_len(page: fitz.Page, footer_ratio: float) -> int:
    footer = _footer_rect(page, footer_ratio)
    try:
        blocks = page.get_text("blocks") or []
    except Exception:
        return 0
    total = 0
    for b in blocks:
        if len(b) < 5:
            continue
        x0, y0, x1, y1, txt = b[0], b[1], b[2], b[3], b[4] or ""
        rect = fitz.Rect(x0, y0, x1, y1)
        if rect.y1 <= footer.y0:
            total += len(txt.strip())
    return total

def _large_image_area_ratio(doc: fitz.Document, page: fitz.Page) -> float:
    try:
        imgs = page.get_images(full=True)
        if not imgs:
            return 0.0
        page_area = page.rect.width * page.rect.height
        max_area = 0.0
        for info in imgs:
            xref = info[0]
            pm = fitz.Pixmap(doc, xref)
            area = pm.width * pm.height
            if area > max_area:
                max_area = area
        return float(max_area) / float(page_area) if page_area > 0 else 0.0
    except Exception:
        return 0.0

def _extract_native_text_json(pdf_path: str, opts: PDFExtractOptions) -> Dict[str, Any]:
    doc = fitz.open(pdf_path)
    page_nums = _iter_pages(doc, opts.page_range)
    pages: List[PageResult] = []
    any_text = False
    ocr_needed_pages: List[int] = []

    for pg in page_nums:
        pr = PageResult(page_number=pg)
        try:
            p = doc.load_page(pg-1)
            txt = p.get_text("text") or ""
            if _has_meaningful_text(txt):
                any_text = True
            pr.text = _postprocess_text(txt.strip())

            pr.links = _extract_links(p)

            if opts.include_images_metadata:
                pr.images = _extract_images_metadata(doc, p)

            if opts.add_footer_artifacts:
                footer = {"logo": "vector", "link": _footer_link(pr.links, p, opts.footer_threshold_ratio)}
                pr.artifacts["footer"] = footer

            # ---- Heuristic: decide OCR candidates per page ----
            non_footer_chars = _non_footer_text_len(p, opts.footer_threshold_ratio)
            big_img_ratio = _large_image_area_ratio(doc, p)
            if (non_footer_chars < opts.min_chars_without_footer) or (big_img_ratio >= opts.min_large_image_area_ratio):
                ocr_needed_pages.append(pg)

        except Exception as e:
            pr.errors.append(f"native_extract_failed: {e}")

        pages.append(pr)

    # Tables
    if opts.detect_tables and (_HAS_CAMELOT or _HAS_PDFPLUMBER):
        tables_all: Dict[int, List[Dict[str, Any]]] = {}
        if "camelot" in opts.table_engine_priority and _HAS_CAMELOT:
            pages_spec = ",".join(str(n) for n in page_nums)
            tables_all = _merge_tables(_tables_via_camelot(pdf_path, pages_spec), tables_all)
        if "pdfplumber" in opts.table_engine_priority and _HAS_PDFPLUMBER:
            tables_all = _merge_tables(tables_all, _tables_via_pdfplumber(pdf_path, page_nums))
        for pr in pages:
            pr.tables = tables_all.get(pr.page_number, [])

    result = {
        "document_type": "pdf",
        "file_name": os.path.basename(pdf_path),
        "page_count": len(page_nums),
        "pages": [asdict(pr) for pr in pages],
        "meta": {
            "ocr_applied": False,
            "has_text": any_text,
            "ocr_candidates": ocr_needed_pages
        }
    }

    # Forms (AcroForm)
    try:
        forms = _extract_form_fields(doc)
        if forms:
            result["meta"]["form_fields"] = forms
    except Exception:
        pass

    return result


# ==============
# OCR Processing
# ==============

def _ocrmypdf_available() -> bool:
    return _which("ocrmypdf")

def _apply_ocrmypdf(in_pdf: str, force: bool = False) -> str:
    """
    Make a searchable PDF. If force=True, use --force-ocr; else --skip-text.
    """
    out = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False).name
    cmd = [
        "ocrmypdf",
        "--rotate-pages",
        "--remove-background",
        "--clean",
        "--fast-web-view",
        "--tesseract-timeout", "0",
    ]
    cmd.append("--force-ocr" if force else "--skip-text")
    cmd.extend([in_pdf, out])
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return out

def _ocr_page_worker(args) -> Tuple[int, str]:
    # Args: (pg, pdf_path, dpi, lang, footer_ratio)
    (pg, pdf_path, dpi, lang, footer_ratio) = args
    doc = fitz.open(pdf_path)
    page = doc.load_page(pg-1)
    # Rasterize
    pix = page.get_pixmap(dpi=dpi)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img = ImageOps.exif_transpose(img)
    # Crop footer off before OCR (reduce URL/page-no noise)
    W, H = img.size
    footer_px = int(H * footer_ratio)
    if footer_px > 0:
        img = img.crop((0, 0, W, max(1, H - footer_px)))
    text = _ocr_image(img, lang)
    return (pg, text)

def _extract_ocr_text_json(pdf_path: str, opts: PDFExtractOptions, only_pages: Optional[List[int]] = None) -> Dict[str, Any]:
    # 1) ocrmypdf pre-pass
    final_pdf = pdf_path
    temp_generated = False
    if opts.use_ocrmypdf and _ocrmypdf_available():
        try:
            final_pdf = _apply_ocrmypdf(pdf_path, force=opts.force_ocr_all)
            temp_generated = True
        except Exception:
            final_pdf = pdf_path
            temp_generated = False

    doc = fitz.open(final_pdf)
    page_nums = _iter_pages(doc, opts.page_range)
    if only_pages:
        only_set = set(only_pages)
        page_nums = [pg for pg in page_nums if pg in only_set]

    # 2) OCR per page (parallel)
    tasks = [(pg, final_pdf, opts.dpi, opts.ocr_lang, opts.footer_threshold_ratio) for pg in page_nums]
    pages_map: Dict[int, str] = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=opts.max_workers) as ex:
        for pg, txt in ex.map(_ocr_page_worker, tasks):
            pages_map[pg] = txt or ""

    pages: List[PageResult] = []
    for pg in page_nums:
        clean_ocr = _postprocess_text(pages_map.get(pg, ""))
        pr = PageResult(page_number=pg, ocr_text=clean_ocr)
        try:
            p = doc.load_page(pg-1)
            pr.links = _extract_links(p)

            if opts.include_images_metadata:
                pr.images = _extract_images_metadata(doc, p)

            if opts.add_footer_artifacts:
                footer = {"logo": "vector", "link": _footer_link(pr.links, p, opts.footer_threshold_ratio)}
                pr.artifacts["footer"] = footer

            if opts.save_footer_snapshot and opts.output_image_dir:
                os.makedirs(opts.output_image_dir, exist_ok=True)
                img = _page_to_image(p, opts.dpi)
                crop = _rasterize_footer_snapshot(img, opts.footer_snapshot_size)
                fname = f"p{pg:02d}_footer.png"
                fpath = os.path.join(opts.output_image_dir, fname)
                crop.save(fpath)

        except Exception as e:
            pr.errors.append(f"ocr_postprocess_failed: {e}")
        pages.append(pr)

    # Tables (run on original pdf path for geometry)
    if opts.detect_tables and (_HAS_CAMELOT or _HAS_PDFPLUMBER):
        tables_all: Dict[int, List[Dict[str, Any]]] = {}
        if "camelot" in opts.table_engine_priority and _HAS_CAMELOT:
            spec = ",".join(str(n) for n in (only_pages or _iter_pages(fitz.open(pdf_path), opts.page_range)))
            tables_all = _merge_tables(_tables_via_camelot(pdf_path, spec), tables_all)
        if "pdfplumber" in opts.table_engine_priority and _HAS_PDFPLUMBER:
            tables_all = _merge_tables(tables_all, _tables_via_pdfplumber(pdf_path, only_pages or _iter_pages(fitz.open(pdf_path), opts.page_range)))
        for pr in pages:
            pr.tables = tables_all.get(pr.page_number, [])

    result = {
        "document_type": "pdf",
        "file_name": os.path.basename(pdf_path),
        "page_count": len(page_nums),
        "pages": [asdict(pr) for pr in pages],
        "meta": {"ocr_applied": True}
    }

    # Forms (AcroForm) from raw PDF
    try:
        with fitz.open(pdf_path) as raw_doc:
            forms = _extract_form_fields(raw_doc)
            if forms:
                result["meta"]["form_fields"] = forms
    except Exception:
        pass

    if temp_generated and not opts.keep_intermediate:
        try:
            os.remove(final_pdf)
        except Exception:
            pass

    return result


# ==================
# Public Entry Point
# ==================

def extract_pdf(pdf_path: str, options: Optional[PDFExtractOptions] = None) -> Dict[str, Any] | str:
    """
    Main entry: robust page-wise extractor.
    Modes:
      - force_ocr_all=True   -> OCR on all pages
      - else native + per-page heuristic OCR for mixed pages
      - fallback to full OCR if native had no text and no candidates detected
    Returns JSON dict by default, or RAW text if options.raw_format=True.
    """
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    opts = options or PDFExtractOptions()

    # A) Force OCR across all pages
    if opts.force_ocr_all:
        ocr_all = _extract_ocr_text_json(pdf_path, opts, only_pages=None)
        return _to_raw(ocr_all) if opts.raw_format else ocr_all

    # B) Native first + heuristic candidates
    native = _extract_native_text_json(pdf_path, opts)
    ocr_candidates: List[int] = native["meta"].get("ocr_candidates", [])
    has_text = bool(native["meta"].get("has_text"))

    if not ocr_candidates and has_text:
        # Native is sufficient
        return _to_raw(native) if opts.raw_format else native

    if ocr_candidates:
        # OCR only candidate pages
        ocr_part = _extract_ocr_text_json(pdf_path, opts, only_pages=ocr_candidates)
        pages_ocr = {p["page_number"]: p for p in ocr_part["pages"]}

        def _merge_and_clean(a: str, b: str) -> str:
            if not a and not b:
                return ""
            la = [ln for ln in (a.splitlines() if a else []) if ln.strip()]
            lb = [ln for ln in (b.splitlines() if b else []) if ln.strip()]
            merged = _dedup_lines(_strip_footer_noise(la + lb))
            s = "\n".join(merged)
            s = _RE_MANY_BLANKS.sub("\n\n", s)
            return s.strip()

        merged_pages = []
        for pr in native["pages"]:
            pg = pr["page_number"]
            o = pages_ocr.get(pg)
            if o:
                final_text = (pr.get("text") or "").strip()
                ocr_text = (o.get("ocr_text") or "").strip()
                combined = _merge_and_clean(final_text, ocr_text)
                pr["text"] = combined
                pr["ocr_text"] = ocr_text
                if not pr.get("links"): pr["links"] = o.get("links", [])
                if not pr.get("images"): pr["images"] = o.get("images", [])
                if not pr.get("artifacts"): pr["artifacts"] = o.get("artifacts", {})
            merged_pages.append(pr)

        native["pages"] = merged_pages
        native["meta"]["ocr_applied"] = True
        return _to_raw(native) if opts.raw_format else native

    # C) Native had no useful text; do full OCR fallback
    ocr_all = _extract_ocr_text_json(pdf_path, opts, only_pages=None)
    return _to_raw(ocr_all) if opts.raw_format else ocr_all


def _to_raw(doc: Dict[str, Any]) -> str:
    lines: List[str] = []
    for p in doc.get("pages", []):
        n = p.get("page_number")
        lines.append(f"===Page {n}===")
        lines.append("{HEADER if any}")
        text = p.get("text") or p.get("ocr_text") or ""
        lines.append(text)
        lines.append("{FOOTER if any}")
        lines.append("")
    return "\n".join(lines)


# =============
# Simple  CLI
# =============

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Robust PDF extractor (JSON or RAW).")
    ap.add_argument("pdf", help="Path to PDF")
    ap.add_argument("--format", choices=["json","raw"], default="json")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--lang", default="eng")
    ap.add_argument("--force-ocr", action="store_true", help="Force OCR on all pages")
    ap.add_argument("--no-ocrmypdf", action="store_true", help="Disable ocrmypdf pre-pass")
    ap.add_argument("--tables", action="store_true", help="Enable table detection (camelot/pdfplumber if installed)")
    ap.add_argument("--footer-snapshot-dir", default=None, help="If set, saves 120x120 footer crops per page")
    ap.add_argument("--page-range", default=None, help="e.g., 1-5")
    ap.add_argument("--min-nonfooter-chars", type=int, default=50, help="Heuristic threshold for non-footer text length")
    ap.add_argument("--min-largeimg-ratio", type=float, default=0.20, help="Heuristic threshold for largest image area / page area")
    args = ap.parse_args()

    pr: Optional[Tuple[int,int]] = None
    if args.page_range:
        m = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", args.page_range)
        if m:
            pr = (int(m.group(1)), int(m.group(2)))

    opts = PDFExtractOptions(
        dpi=args.dpi,
        ocr_lang=args.lang,
        use_ocrmypdf=(not args.no_ocrmypdf),
        detect_tables=args.tables,
        output_image_dir=args.footer_snapshot_dir,
        save_footer_snapshot=bool(args.footer_snapshot_dir),
        page_range=pr,
        raw_format=(args.format == "raw"),
        force_ocr_all=args.force_ocr,
        min_chars_without_footer=args.min_nonfooter_chars,
        min_large_image_area_ratio=args.min_largeimg_ratio,
    )

    out = extract_pdf(args.pdf, opts)
    if isinstance(out, str):
        print(out)
    else:
        print(json.dumps(out, ensure_ascii=False, indent=2))
