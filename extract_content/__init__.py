# extract_content/__init__.py
import azure.functions as func
import io, os, re, json, uuid, csv, tempfile, shutil, sys, time
import fitz  # PyMuPDF
from PIL import Image, ImageOps, ImageFilter
import pytesseract
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import cv2
from transformers import AutoProcessor, AutoModelForCausalLM
import torch
def _blue_ink_binary(pil_img: Image.Image) -> np.ndarray:
    """Mavi mürekkebi vurgulayan ikili görüntü döndürür (0/255)."""
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    # 2.2x büyüt: küçük yazıda LSTM performansı artar
    bgr = cv2.resize(bgr, None, fx=2.2, fy=2.2, interpolation=cv2.INTER_CUBIC)

    # Mavi–Kırmızı farkı: mavi mürekkebi öne çıkar
    B, G, R = cv2.split(bgr)
    ink = cv2.subtract(B, R)                    # mavi >> kırmızı ise yüksek
    ink = cv2.normalize(ink, None, 0, 255, cv2.NORM_MINMAX)

    # Gürültü azalt: bilateral + median
    ink = cv2.bilateralFilter(ink, 7, 50, 50)
    ink = cv2.medianBlur(ink, 3)

    # Adaptif eşik (mürekkebi beyaz yapalım)
    bin_ink = cv2.adaptiveThreshold(ink, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 35, -5)

    # Defter yatay çizgilerini yakala ve çıkar
    h, w = bin_ink.shape
    k = max(15, w // 22)                        # genişliğe göre kernel
    horiz_k = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))
    # Çizgiler beyaz ise invert edip açma uygula
    inv = 255 - bin_ink
    lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, horiz_k, iterations=1)
    # Çizgileri maske olarak kullanıp orijinalden çıkar
    cleaned = cv2.bitwise_and(bin_ink, 255 - lines)

    # Tesseract için siyah üstüne beyaz metin (invert)
    return 255 - cleaned

def _preprocess_handwriting_cv(pil_img: Image.Image) -> Image.Image:
    bin_img = _blue_ink_binary(pil_img)
    # Küçük açıklıkları kapat (harfler bütünleşsin)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=1)
    return Image.fromarray(bin_img).convert("L")

def _cv_from_pil(pil_img: Image.Image) -> np.ndarray:
    # PIL (RGB) -> OpenCV (BGR)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def _preprocess_handwriting_cv(pil_img: Image.Image, fast: bool = False) -> Image.Image:
    """
    Çizgili defterde mavi mürekkebi öne çıkartıp yatay çizgileri bastırır.
    Optimized for speed: reduced resize, simplified HSV, fewer iterations.
    """
    img = _cv_from_pil(pil_img)

    # 1) 2x büyüt (LSTM küçük yazıda zorlanır)
    img = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    # 2) HSV ile mavi tonlarını vurgula (mürekkep mavi)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 40, 40])    # alt sınır (H,S,V)
    upper_blue = np.array([140, 255, 255]) # üst sınır
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # 3) Gri + kontrast
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # 4) Yatay çizgileri (defter çizgileri) morfoloji ile bastır
    #    genişlik bazlı yatay kernel
    w = gray.shape[1]
    k = max(15, w // 25)  # çizgi kalınlığı ~ genişliğe bağlı
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))
    # İnce çizgileri yakalamak için adaptif threshold
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 35, 15)
    # Sadece yatay çizgileri çıkar
    horiz = cv2.morphologyEx(thr, cv2.MORPH_OPEN, horiz_kernel, iterations=1)
    # Çizgileri orijinalden çıkar
    no_lines = cv2.bitwise_and(thr, cv2.bitwise_not(horiz))

    # 5) Mavi maskesini de ekleyip yazıyı güçlendir
    ink = cv2.bitwise_or(no_lines, mask_blue)

    # 6) Median + küçük açma ile pürüz azalt - skip if fast
    if not fast:
        ink = cv2.medianBlur(ink, 3)
        small_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, small_kernel, iterations=1)

    # 7) Tesseract için siyah üstüne beyaz metin (invert)
    bin_img = cv2.bitwise_not(ink)

    # OpenCV -> PIL (L) gri
    pil_out = Image.fromarray(bin_img).convert("L")
    return pil_out

def _tesseract_lang_available(lang: str) -> bool:
    try:
        langs = pytesseract.get_languages(config="")
        return lang in langs
    except Exception:
        # bazı ortamlarda listeleme çalışmayabiliyor
        return (lang == "eng")

# --- Tesseract yolu (Windows) ---
TESS_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(TESS_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESS_PATH

# --- Ayarlar / Heuristikler ---
FOOTER_RATIO = 0.08
OCR_DPI      = 400
NONFOOTER_MIN_CHARS = 30
FORCE_OCR_PAGES = set()

# --- Regex'ler ---
RE_FOOTER_LINE   = re.compile(r"^\s*(https?://\S+)\s*page\s+\d+\s+of\s+\d+\s*$", re.I)
RE_MANY_NEWLINES = re.compile(r"\n{3,}")
RE_MANY_SPACES   = re.compile(r"[ \t]{2,}")
RE_COPY_MARK     = re.compile(r"[©®]")
RE_VISUAL_PIPE   = re.compile(r"\|")
RE_GARBAGE_LINE  = re.compile(r"(?m)^\s*(?:[+•·oO]|—|–|_#|,|;|:|\.)\s*$|^\s*(?:ox|xo|oxo)\*?\s*$")
_BOUNDARY_RE     = re.compile(r'boundary=(?:"([^"]+)"|([^;]+))', re.I)

# ---------- Yardımcılar ----------
def _strip_footer_and_garbage(text: str) -> str:
    if not text: return ""
    kept = []
    for ln in text.splitlines():
        s = (ln or "").strip()
        if not s: kept.append(""); continue
        if RE_FOOTER_LINE.match(s): continue
        if RE_GARBAGE_LINE.match(s): continue
        kept.append(s)
    out = "\n".join(kept)
    out = RE_MANY_NEWLINES.sub("\n\n", out)
    return out.strip()

def _normalize_text(text: str) -> str:
    if not text: return ""
    text = RE_COPY_MARK.sub("", text)
    text = RE_VISUAL_PIPE.sub("I", text)
    text = RE_MANY_SPACES.sub(" ", text)
    lines = [ln.strip() for ln in text.splitlines()]
    text = "\n".join(lines)
    text = _strip_footer_and_garbage(text)
    return text

def _non_footer_text_len(page: fitz.Page) -> int:
    r = page.rect; footer_y0 = r.y1 - r.height * FOOTER_RATIO
    total = 0
    try:
        blocks = page.get_text("blocks") or []
        for b in blocks:
            if len(b) < 5: continue
            x0, y0, x1, y1, txt = b[0], b[1], b[2], b[3], b[4] or ""
            if y1 <= footer_y0: total += len((txt or "").strip())
    except Exception: pass
    return total
def _preprocess_handwriting(img: Image.Image) -> Image.Image:
    """
    Defter çizgilerini baskılamak ve mavi mürekkebi öne çıkarmak için hafif bir ön-işleme.
    OpenCV yoksa PIL ile en pratik yaklaşım: ölçekleme + autocontrast + median + threshold.
    """
    # 1) 2x büyüt (Tesseract LSTM küçük yazıda zorlanır)
    w, h = img.size
    img = img.resize((w*2, h*2))
    # 2) Gri ve kontrast
    g = ImageOps.grayscale(img)
    g = ImageOps.autocontrast(g)
    # 3) İnce yatay çizgileri azaltmak için median
    g = g.filter(ImageFilter.MedianFilter(size=3))
    # 4) Yumuşak ikili eşik (defter çizgilerini hafif bastırır, mürekkebi korur)
    g = g.point(lambda p: 255 if p > 180 else 0)
    # 5) Hafif keskinleştirme
    g = g.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=2))
    return g

def _preprocess_for_ocr(img: Image.Image) -> Image.Image:
    g = ImageOps.grayscale(img)
    g = ImageOps.autocontrast(g)  # düzeltme: 'g' üzerinde autocontrast
    g = g.point(lambda p: 255 if p > 127 else 0)
    g = g.filter(ImageFilter.UnsharpMask(radius=2, percent=160, threshold=3))
    return g

def _ocr_page_image(img: Image.Image, lang: str = "eng", psm=6, fast: bool = False) -> str:
    g = _preprocess_for_ocr(img)
    if not _tesseract_lang_available(lang):
        lang = "eng"
    best, best_len = "", 0
    psms = [psm] if fast else [psm, 3, 4]
    for p in psms:  # mevcut psm + fallback
        try:
            tt = pytesseract.image_to_string(g, lang=lang, config=f"--oem 3 --psm {p}") or ""
        except Exception:
            tt = ""
        clen = len(("".join(tt.split())) if tt else "")
        if clen > best_len:
            best, best_len = tt, clen
    return (best or "").strip()


def _collect_links(page: fitz.Page):
    out = []
    try:
        for ln in page.get_links() or []:
            uri = ln.get("uri")
            if uri: out.append({"target": uri})
    except Exception: pass
    return out

def _footer_snapshot(page: fitz.Page, page_no: int) -> dict | None:
    return None

def _detect_footer_artifacts(links: list) -> dict:
    footer = {}
    if links: footer["link"] = links[0].get("target")
    footer["logo"] = "vector"
    return footer

def _collect_form_fields(page: fitz.Page):
    fields = []
    try:
        widgets = page.widgets()
        for w in widgets:
            name  = getattr(w, "field_name", None) or getattr(w, "name", None)
            ftype = getattr(w, "field_type", None)
            value = getattr(w, "field_value", None)
            if value is None: value = getattr(w, "value", None)
            rect  = getattr(w, "rect", None)
            bbox  = [rect.x0, rect.y0, rect.x1, rect.y1] if rect is not None else None
            fields.append({"name": name, "value": value})
    except Exception: pass
    return fields

def _page_text_or_ocr(doc_bytes: bytes, page_index_0: int, force_ocr: bool, lang: str, ocrdpi: int, fast: bool):
    t0 = time.perf_counter()
    with fitz.open(stream=io.BytesIO(doc_bytes), filetype="pdf") as doc:
        page = doc.load_page(page_index_0)
        nonfooter_len = _non_footer_text_len(page)
        need_ocr = ((not fast and (nonfooter_len < NONFOOTER_MIN_CHARS)) or force_ocr or ((page_index_0 + 1) in FORCE_OCR_PAGES))

        # Use pdfplumber for text extraction and OCR
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(doc_bytes)) as pdf:
                pl_page = pdf.pages[page_index_0]
                native = pl_page.extract_text() or ""
                native_stripped = native.strip()
                if not need_ocr and len(native_stripped) >= NONFOOTER_MIN_CHARS:
                    links = _collect_links(page)
                    footer_img = _footer_snapshot(page, page_index_0 + 1)
                    elapsed_ms = int((time.perf_counter() - t0) * 1000)
                    return {
                        "text": native,
                        "ocr_text": _normalize_text(native),
                        "links": links,
                        "footer_img": footer_img,
                        "elapsed_ms": elapsed_ms
                    }

                # OCR
                image = pl_page.to_image(resolution=ocrdpi)
                pil_img = image.original
                ocr_text = pytesseract.image_to_string(pil_img, lang=lang) or ""
        except Exception:
            # Fallback to fitz for text and OCR
            native = page.get_text("text") or ""
            native_stripped = native.strip()
            if not need_ocr and len(native_stripped) >= NONFOOTER_MIN_CHARS:
                links = _collect_links(page)
                footer_img = _footer_snapshot(page, page_index_0 + 1)
                elapsed_ms = int((time.perf_counter() - t0) * 1000)
                return {
                    "text": native,
                    "ocr_text": _normalize_text(native),
                    "links": links,
                    "footer_img": footer_img,
                    "elapsed_ms": elapsed_ms
                }

            # Fallback OCR
            pix = page.get_pixmap(dpi=ocrdpi, alpha=False, colorspace=fitz.csGRAY)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            best, best_len = "", 0
            psms = [6] if fast else [6, 4]
            for psm in psms:
                t = _ocr_page_image(img, lang=lang, psm=psm, fast=fast)
                clen = len(re.sub(r"\s+", "", t))
                if clen > best_len: best, best_len = t, clen
            ocr_text = best
            native = native  # already set

        merged, seen = [], set()
        for ln in (native.splitlines() + ocr_text.splitlines()):
            s = (ln or "").strip()
            if s == "":
                merged.append("")
                continue
            if s not in seen:
                merged.append(s); seen.add(s)
        cleaned = _normalize_text("\n".join(merged))
        links = _collect_links(page)
        footer_img = _footer_snapshot(page, page_index_0 + 1)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        return {
            "text": native,
            "ocr_text": cleaned,
            "links": links,
            "footer_img": footer_img,
            "elapsed_ms": elapsed_ms
        }

# ---------- Tür kontrol & çıkarıcılar ----------
def _is_pdf(fname, ctype):  return (ctype == "application/pdf") or fname.lower().endswith(".pdf")
def _is_docx(fname, ctype): return fname.lower().endswith(".docx") or (ctype or "") in {"application/vnd.openxmlformats-officedocument.wordprocessingml.document"}
def _is_pptx(fname, ctype): return fname.lower().endswith(".pptx") or (ctype or "") in {"application/vnd.openxmlformats-officedocument.presentationml.presentation"}
def _is_txt(fname, ctype):  return fname.lower().endswith(".txt") or (ctype or "").startswith("text/")
def _is_image(fname, ctype):
    ext = os.path.splitext(fname.lower())[1]
    return ext in {".png",".jpg",".jpeg",".tif",".tiff"} or ((ctype or "").startswith("image/"))
def _is_html(fname, ctype):
    ext = os.path.splitext(fname.lower())[1]
    return ext in {".html",".htm"} or (ctype or "") in {"text/html","application/xhtml+xml"}
def _is_rtf(fname, ctype):
    ext = os.path.splitext(fname.lower())[1]
    return ext == ".rtf" or (ctype or "") in {"application/rtf","text/rtf"}
def _is_excel(fname, ctype):
    ext = os.path.splitext(fname.lower())[1]
    return ext in {".xlsx",".xlsm",".xls",".xlt",".xltx"} or (ctype or "").startswith("application/vnd.ms-excel") or "spreadsheet" in (ctype or "")
def _is_csv(fname, ctype):
    ext = os.path.splitext(fname.lower())[1]
    return ext == ".csv" or (ctype or "") in {"text/csv","application/csv"}

# ---- Word / PPTX / TXT / IMG / HTML / RTF ----
def _extract_docx_bytes(data: bytes) -> dict:
    try: import docx
    except Exception as e: return {"errors":[f"python-docx not available: {e}"]}
    try: d = docx.Document(io.BytesIO(data))
    except Exception as e: return {"errors":[f"Failed to open DOCX: {e}"]}

    paras = [p.text.strip() for p in d.paragraphs if (p.text or "").strip()]
    table_lines = []
    for tbl in d.tables:
        for row in tbl.rows:
            cells = [ (c.text or "").strip() for c in row.cells ]
            table_lines.append("\t".join(cells))
    content = _normalize_text("\n".join(paras + ([""] if paras and table_lines else []) + table_lines).strip())
    page = {"page_number": 1, "text": content, "ocr_text": content, "links": [], "images": [], "artifacts": {}}
    return {"page_count": 1, "pages": [page], "errors": []}

def _extract_pptx_bytes(data: bytes) -> dict:
    try: from pptx import Presentation
    except Exception as e: return {"errors":[f"python-pptx not available: {e}"]}
    try: prs = Presentation(io.BytesIO(data))
    except Exception as e: return {"errors":[f"Failed to open PPTX: {e}"]}

    def _shape_texts(shape):
        lines = []
        try:
            if hasattr(shape,"has_text_frame") and shape.has_text_frame and shape.text_frame:
                for p in shape.text_frame.paragraphs:
                    txt = "".join([run.text or "" for run in p.runs]) or (p.text or "")
                    if txt: lines.append(txt)
            if hasattr(shape,"has_table") and shape.has_table and shape.table:
                for row in shape.table.rows:
                    cells = []
                    for cell in row.cells:
                        ctext = " ".join((cell.text or "").split())
                        cells.append(ctext.strip())
                    lines.append("\t".join(cells))
            if hasattr(shape,"shapes"):
                for shp in shape.shapes:
                    lines.extend(_shape_texts(shp))
        except Exception: pass
        return lines

    pages = []
    for idx, slide in enumerate(prs.slides, start=1):
        lines = []
        for shp in slide.shapes:
            lines.extend(_shape_texts(shp))
        try:
            if slide.has_notes_slide and slide.notes_slide and slide.notes_slide.notes_text_frame:
                note = slide.notes_slide.notes_text_frame.text or ""
                if note.strip():
                    lines.append("")
                    lines.append("[Notes]")
                    lines.append(note.strip())
        except Exception: pass
        content = _normalize_text("\n".join([ln for ln in lines if ln is not None]).strip())
        pages.append({
            "page_number": idx, "text": content, "ocr_text": content,
            "links": [], "images": [], "artifacts": {"slide_layout": getattr(getattr(slide,"slide_layout",None),"name",None)}
        })
    return {"page_count": len(pages), "pages": pages, "errors": []}

def _extract_txt_bytes(data: bytes, encoding_guess="utf-8") -> dict:
    try:
        try: text = data.decode(encoding_guess, errors="replace")
        except Exception: text = data.decode("latin-1", errors="replace")
    except Exception as e: return {"errors":[f"Failed to decode TXT: {e}"]}
    text = _normalize_text(text.strip())
    page = {"page_number": 1, "text": text, "ocr_text": text, "links": [], "images": [], "artifacts": {}}
    return {"page_count": 1, "pages": [page], "errors": []}

def _extract_image_bytes_with_ocr(data: bytes, lang="eng") -> dict:
    try:
        img = Image.open(io.BytesIO(data))
    except Exception as e:
        return {"errors":[f"Failed to open image: {e}"]}

    try:
        device_id = 0 if torch.cuda.is_available() else -1
        device = f"cuda:{device_id}" if device_id >= 0 else "cpu"

        processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True, torch_dtype=torch.float32)
        model.to(device)
        model.eval()

        prompt = "<OCR>"
        inputs = processor(text=prompt, images=img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
            )
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Clean up the output
        text = generated_text.strip()
        if text.startswith("<OCR>"):
            text = text[5:].strip()
        text = text.replace("<pad>", "").replace("</s>", "").strip()
        t = _normalize_text(text)
    except Exception as e:
        return {"errors":[f"OCR failed: {e}"]}

    page = {"page_number": 1, "text": t, "ocr_text": t, "links": [], "images": [], "artifacts": {
        "debug": {"lang_used": lang, "engine": "florence-2", "note": "img pipeline"}
    }}
    return {"page_count": 1, "pages": [page], "errors": []}




def _extract_html_bytes(data: bytes) -> dict:
    try: from bs4 import BeautifulSoup
    except Exception as e: return {"errors":[f"beautifulsoup4 not available: {e}"]}
    try: text = data.decode("utf-8", errors="replace")
    except Exception: text = data.decode("latin-1", errors="replace")
    try: soup = BeautifulSoup(text, "lxml")
    except Exception: soup = BeautifulSoup(text, "html.parser")

    # Güvenli script/style/noscript temizliği
    for el in soup.find_all(["script", "style", "noscript"]):
        try:
            el.decompose()
        except Exception:
            try:
                el.extract()
            except Exception:
                pass

    extracted = soup.get_text(separator="\n")
    extracted = "\n".join([ln.strip() for ln in extracted.splitlines() if ln.strip()])
    extracted = _normalize_text(extracted)
    links = []
    try:
        for a in soup.find_all("a", href=True):
            href = a.get("href")
            if href: links.append({"target": href})
    except Exception: pass
    page = {"page_number": 1, "text": extracted, "ocr_text": extracted, "links": links, "images": [], "artifacts": {"title": soup.title.string.strip() if soup.title and soup.title.string else None}}
    return {"page_count": 1, "pages": [page], "errors": []}

def _extract_rtf_bytes(data: bytes) -> dict:
    try: from striprtf.striprtf import rtf_to_text
    except Exception as e: return {"errors":[f"striprtf not available: {e}"]}
    try: txt = rtf_to_text(data.decode("latin-1", errors="ignore"))
    except Exception: txt = rtf_to_text(data.decode("utf-8", errors="ignore"))
    txt = _normalize_text(txt)
    page = {"page_number": 1, "text": txt, "ocr_text": txt, "links": [], "images": [], "artifacts": {}}
    return {"page_count": 1, "pages": [page], "errors": []}

# ---- Excel / CSV ----
def _extract_excel_bytes(data: bytes, filename: str) -> dict:
    ext = os.path.splitext(filename.lower())[1]
    if ext in {".xlsx", ".xlsm", ".xltx"}:
        try: import openpyxl
        except Exception as e: return {"errors":[f"openpyxl not available: {e}"]}
        try: wb = openpyxl.load_workbook(io.BytesIO(data), data_only=True, read_only=True)
        except Exception as e: return {"errors":[f"Failed to open Excel: {e}"]}
        pages = []
        for idx, ws in enumerate(wb.worksheets, start=1):
            rows_txt = []
            try:
                for row in ws.iter_rows(values_only=True):
                    cells = ["" if v is None else str(v) for v in row]
                    rows_txt.append("\t".join(cells))
            except Exception: pass
            content = _normalize_text("\n".join(rows_txt).strip())
            pages.append({"page_number": idx, "text": content, "ocr_text": content, "links": [], "images": [], "artifacts": {"sheet": ws.title}})
        return {"page_count": len(pages), "pages": pages, "errors": []}
    elif ext in {".xls", ".xlt"}:
        try: import xlrd
        except Exception as e: return {"errors":[f"xlrd not available for .xls: {e}"]}
        try: wb = xlrd.open_workbook(file_contents=data)
        except Exception as e: return {"errors":[f"Failed to open XLS: {e}"]}
        pages = []
        for idx in range(wb.nsheets):
            sh = wb.sheet_by_index(idx)
            rows_txt = []
            for r in range(sh.nrows):
                vals = []
                for c in range(sh.ncols):
                    v = sh.cell_value(r, c)
                    vals.append("" if v is None else str(v))
                rows_txt.append("\t".join(vals))
            content = _normalize_text("\n".join(rows_txt).strip())
            pages.append({"page_number": idx+1, "text": content, "ocr_text": content, "links": [], "images": [], "artifacts": {"sheet": sh.name}})
        return {"page_count": len(pages), "pages": pages, "errors": []}
    else:
        return {"errors":[f"Unsupported Excel extension: {ext}"]}

def _extract_csv_bytes(data: bytes) -> dict:
    try: txt = data.decode("utf-8", errors="replace")
    except Exception: txt = data.decode("latin-1", errors="replace")
    rows = []
    try:
        reader = csv.reader(io.StringIO(txt))
        for row in reader: rows.append("\t".join(row))
    except Exception:
        rows = [ln for ln in txt.splitlines()]
    content = _normalize_text("\n".join(rows).strip())
    page = {"page_number": 1, "text": content, "ocr_text": content, "links": [], "images": [], "artifacts": {"dialect": "csv"}}
    return {"page_count": 1, "pages": [page], "errors": []}

# ---------- OCRmyPDF (opsiyonel prepass) ----------
def _try_ocrmypdf(input_bytes: bytes) -> bytes | None:
    try:
        import subprocess
        with tempfile.TemporaryDirectory() as td:
            inp = os.path.join(td, "in.pdf")
            outp = os.path.join(td, "out.pdf")
            with open(inp, "wb") as f: f.write(input_bytes)
            cmd = ["ocrmypdf", "--skip-text", "--optimize", "1", "--tesseract-timeout", "0", inp, outp]
            r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
            if r.returncode != 0:
                return None
            with open(outp, "rb") as f:
                return f.read()
    except Exception:
        return None

# ---------- PDF Tablo çıkarımı (opsiyonel) ----------
def _extract_tables_from_page(doc_bytes: bytes, page_num_1: int, engine: str):
    tables = []
    if engine == "camelot":
        try:
            import camelot
            with tempfile.TemporaryDirectory() as td:
                p = os.path.join(td, "tmp.pdf")
                with open(p, "wb") as f: f.write(doc_bytes)
                ts = camelot.read_pdf(p, flavor="lattice", pages=str(page_num_1))
                for t in ts:
                    rows = [list(map(str, r)) for r in t.data]
                    tables.append({"page": page_num_1, "rows": rows})
        except Exception:
            pass
    elif engine == "plumber":
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(doc_bytes)) as pdf:
                p = pdf.pages[page_num_1-1]
                tbs = p.extract_tables() or []
                for tb in tbs:
                    rows = [[("" if c is None else str(c)).strip() for c in row] for row in tb]
                    tables.append({"page": page_num_1, "rows": rows})
        except Exception:
            pass
    return {"tables": tables}

# ---------- PDF resimleri çıkar ----------
def _extract_images_from_page(doc_bytes: bytes, page_num_1: int):
    return []

# ---------- Çoklu multipart okuma ----------
def _parse_multipart(req: func.HttpRequest, field_names=("files", "file")):
    ctype = (req.headers.get("Content-Type") or req.headers.get("content-type") or "").strip()
    clow = ctype.lower()
    if "multipart/form-data" not in clow:
        raise ValueError("Send as multipart/form-data with key 'files' (or 'file').")

    m = _BOUNDARY_RE.search(ctype)
    boundary = (m.group(1) or m.group(2)).strip() if m else None
    if not boundary: raise ValueError("multipart boundary missing.")
    bbytes = boundary.encode("utf-8")

    body = req.get_body()
    parts = None
    for delim in (b"--"+bbytes, bbytes):
        tmp = body.split(delim)
        if len(tmp) > 1: parts = tmp; break
    if parts is None: raise ValueError("Could not split multipart body with provided boundary.")

    files = []
    for p in parts:
        if not p or p in (b"--\r\n", b"--", b"\r\n"): continue
        if p.startswith(b"\r\n"): p = p[2:]
        p = p.rstrip(b"\r\n")
        if p.endswith(b"--"): p = p[:-2]
        if b"\r\n\r\n" not in p: continue
        header, data = p.split(b"\r\n\r\n", 1)
        data = data.rstrip(b"\r\n")

        htxt = header.decode(errors="ignore")
        hlow = htxt.lower()

        want = False
        for nm in field_names:
            if re.search(rf'name=(?:"|\')?{re.escape(nm)}(?:"|\')?', hlow):
                want = True; break
        if not want: continue

        mfn = re.search(r'filename="([^"]*)"', htxt)
        filename = os.path.basename(mfn.group(1)) if mfn else None

        mct = re.search(r"content-type:\s*([^\r\n;]+)", hlow)
        content_type = mct.group(1).strip() if mct else None

        if data: files.append({"filename": filename, "content_type": content_type, "data": data})

    if not files: raise ValueError("No file parts found under 'files' or 'file'.")
    return files

# ---------- Azure Function ----------
def main(req: func.HttpRequest) -> func.HttpResponse:
    """
    Query:
      - ?forceocr=true|false
      - ?fast=1
      - ?ocrdpi=320
      - ?lang=eng|tur|...
      - ?pages=1-3,5
      - ?workers=4
      - ?preocr=1
      - ?tables=camelot|plumber
      - ?tables_out=csv|json|both
      - ?images=1   (PDF embeeded raster'ları artifacts klasörüne çıkar)
    """
    try:
        t0_total = time.perf_counter()

        # Query params
        forceocr = False; fast = False; lang = "eng"; pages_arg = None; ocrdpi = OCR_DPI
        workers = None; preocr = False; tables_engine = None; tables_out = None; want_images = False
        if "?" in req.url:
            from urllib.parse import parse_qs
            qs = parse_qs(req.url.split("?", 1)[1])
            forceocr = str(qs.get("forceocr", ["false"])[0]).lower() in {"1","true","yes"}
            fast     = str(qs.get("fast", ["0"])[0]).lower() in {"1","true","yes"}
            lang     = (qs.get("lang", ["eng"])[0] or "eng")
            pages_arg= qs.get("pages", [None])[0]
            preocr   = str(qs.get("preocr", ["0"])[0]).lower() in {"1","true","yes"}
            tables_engine = (qs.get("tables", [None])[0] or None)
            tables_out = (qs.get("tables_out", [None])[0] or None)
            want_images = str(qs.get("images", ["0"])[0]).lower() in {"1","true","yes"}
            try: ocrdpi = int(qs.get("ocrdpi", [str(OCR_DPI)])[0])
            except Exception: ocrdpi = OCR_DPI
            try:
                w = qs.get("workers", [None])[0]
                workers = int(w) if w else None
            except Exception:
                workers = None

        def _expand_pages(pages_arg, total):
            if not pages_arg: return list(range(total))
            wanted = set()
            for part in str(pages_arg).split(","):
                part = (part or "").strip()
                if not part: continue
                if "-" in part:
                    a,b = part.split("-",1); a = max(1,int(a)); b = min(total,int(b))
                    if a <= b: wanted.update(range(a-1, b))
                else:
                    idx = max(1, int(part)) - 1
                    if 0 <= idx < total: wanted.add(idx)
            return sorted(wanted)

        files = _parse_multipart(req, field_names=("files", "file"))
        if len(files) != 1:
            return func.HttpResponse("Single file endpoint: exactly one file required.", status_code=400)

        f = files[0]
        fname = f.get("filename") or f"tmp_{uuid.uuid4().hex[:8]}"
        ctype = (f.get("content_type") or "").lower()
        data = f.get("data") or b""
        file_size_mb = len(data) / (1024 * 1024)

        # Validation
        if file_size_mb > 50:
            return func.HttpResponse(f"File size exceeds 50 MB limit for '{fname}'.", status_code=400)

        if _is_pdf(fname, ctype):
            try:
                with fitz.open(stream=io.BytesIO(data), filetype="pdf") as doc_tmp:
                    total_pages = doc_tmp.page_count
            except Exception:
                return func.HttpResponse(f"'{fname}' is invalid PDF file.", status_code=400)
            if total_pages > 300:
                return func.HttpResponse(f"File '{fname}' has more than 300 pages.", status_code=400)

        out_results = []

        # Process the single file
        file_result = {"filename": fname, "mimetype": ctype or None, "errors": []}

        try:
                if _is_pdf(fname, ctype):
                    doc_bytes = data
                    if preocr:
                        pre = _try_ocrmypdf(data)
                        if pre: doc_bytes = pre

                    try:
                        with fitz.open(stream=io.BytesIO(doc_bytes), filetype="pdf") as doc_tmp:
                            total_pages = doc_tmp.page_count
                    except Exception as e:
                        file_result["errors"].append(f"PDF open/process failed: {e}")
                        out_results.append(file_result)
                        result = {"results": out_results}
                        resp = func.HttpResponse(json.dumps(result, ensure_ascii=False, indent=2),
                                                 mimetype="application/json; charset=utf-8")
                        total_ms = int((time.perf_counter() - t0_total) * 1000)
                        resp.headers["X-Debug-Parts"] = str(len(out_results))
                        resp.headers["X-Perf-TotalMs"] = str(total_ms)
                        resp.headers["X-Perf-Workers"] = str(workers or os.cpu_count() or 2)
                        resp.headers["X-Perf-OCRDPI"] = str(ocrdpi)
                        return resp

                    indices = _expand_pages(pages_arg, total_pages)
                    pages_json = []

                    max_workers = workers or os.cpu_count() or 2
                    tasks = {}
                    with ThreadPoolExecutor(max_workers=max_workers) as ex:
                        for i in indices:
                            fut = ex.submit(_page_text_or_ocr, doc_bytes, i, forceocr, lang, ocrdpi, fast)
                            tasks[fut] = i
                        for fut in as_completed(tasks):
                            i = tasks[fut]
                            try:
                                res = fut.result()
                            except Exception as e:
                                res = {"text":"", "ocr_text":"", "links":[], "footer_img":None, "elapsed_ms": None}
                                file_result["errors"].append(f"Page {i+1} error: {e}")
                            try:
                                with fitz.open(stream=io.BytesIO(doc_bytes), filetype="pdf") as dd:
                                    p = dd.load_page(i)
                                    form_fields = _collect_form_fields(p)
                                    artifacts = {"footer": _detect_footer_artifacts(res.get("links", []))}
                            except Exception:
                                form_fields = []; artifacts = {"footer": {}}

                            page_obj = {
                                "page_number": i + 1,
                                "text": res.get("text",""),
                                "ocr_text": res.get("ocr_text",""),
                                "links": res.get("links", []),
                                "images": [res.get("footer_img")] if res.get("footer_img") else [],
                                "artifacts": artifacts,
                                "form_fields": form_fields,
                                "perf_ms": res.get("elapsed_ms")
                            }

                            if want_images:
                                imgs = _extract_images_from_page(doc_bytes, i+1)
                                page_obj.setdefault("images", []).extend(imgs)

                            pages_json.append(page_obj)

                    pages_json.sort(key=lambda x: x["page_number"])

                    if tables_engine in {"camelot","plumber"}:
                        for pj in pages_json:
                            tinfo = _extract_tables_from_page(doc_bytes, pj["page_number"], tables_engine)
                            tables = tinfo.get("tables", [])
                            if tables:
                                if tables_out in {"json", "both"}:
                                    pj["tables"] = tables

                    file_result.update({"page_count": total_pages, "pages": pages_json})

                elif _is_pptx(fname, ctype):
                    res = _extract_pptx_bytes(data)
                    if res.get("errors"): file_result["errors"].extend(res["errors"])
                    else: file_result.update({"page_count": res["page_count"], "pages": res["pages"]})

                elif _is_docx(fname, ctype):
                    res = _extract_docx_bytes(data)
                    if res.get("errors"): file_result["errors"].extend(res["errors"])
                    else: file_result.update({"page_count": res["page_count"], "pages": res["pages"]})

                elif _is_excel(fname, ctype):
                    res = _extract_excel_bytes(data, fname)
                    if res.get("errors"): file_result["errors"].extend(res["errors"])
                    else: file_result.update({"page_count": res["page_count"], "pages": res["pages"]})

                elif _is_csv(fname, ctype):
                    res = _extract_csv_bytes(data)
                    if res.get("errors"): file_result["errors"].extend(res["errors"])
                    else: file_result.update({"page_count": res["page_count"], "pages": res["pages"]})

                elif _is_html(fname, ctype):
                    res = _extract_html_bytes(data)
                    if res.get("errors"): file_result["errors"].extend(res["errors"])
                    else: file_result.update({"page_count": res["page_count"], "pages": res["pages"]})

                elif _is_rtf(fname, ctype):
                    res = _extract_rtf_bytes(data)
                    if res.get("errors"): file_result["errors"].extend(res["errors"])
                    else: file_result.update({"page_count": res["page_count"], "pages": res["pages"]})

                elif _is_txt(fname, ctype):
                    res = _extract_txt_bytes(data)
                    if res.get("errors"): file_result["errors"].extend(res["errors"])
                    else: file_result.update({"page_count": res["page_count"], "pages": res["pages"]})

                elif _is_image(fname, ctype):
                    res = _extract_image_bytes_with_ocr(data, lang=lang)
                    if res.get("errors"): file_result["errors"].extend(res["errors"])
                    else: file_result.update({"page_count": res["page_count"], "pages": res["pages"]})

                else:
                    file_result["errors"].append("Not yet implemented (other formats).")

        except Exception as e:
            file_result["errors"].append(f"Router error: {e}")

        out_results.append(file_result)

        result = {"results": out_results}
        resp = func.HttpResponse(json.dumps(result, ensure_ascii=False, indent=2),
                                 mimetype="application/json; charset=utf-8")
        # Perf header'ları
        total_ms = int((time.perf_counter() - t0_total) * 1000)
        resp.headers["X-Debug-Parts"] = str(len(out_results))
        resp.headers["X-Perf-TotalMs"] = str(total_ms)
        resp.headers["X-Perf-Workers"] = str(workers or os.cpu_count() or 2)
        resp.headers["X-Perf-OCRDPI"] = str(ocrdpi)
        return resp

    except ValueError as ve:
        return func.HttpResponse(str(ve), status_code=400)
    except Exception as e:
        return func.HttpResponse(f"Extraction failed: {e}", status_code=500)
