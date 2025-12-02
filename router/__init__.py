import azure.functions as func
import importlib
from urllib.parse import parse_qs

# Not: extract_content (JSON) ve extract_content_raw (TEXT) fonksiyonları zaten mevcut.
# Burada sadece mode'a göre ilgili fonksiyonun main(req)'ini çağırıyoruz.

def main(req: func.HttpRequest) -> func.HttpResponse:
    """
    /api/router?mode=json|raw[&fast=1][&forceocr=1][&lang=eng][&pages=1-3,5][&ocrdpi=300]
    - mode=json  -> extract_content
    - mode=raw   -> extract_content_raw

    Hız (performance) ile ilgili parametreler:
      - fast=1        : OCR'ı zorlamaz, native metin yeterliyse OCR’a hiç gitmez.
      - forceocr=1    : Özellikle zayıf PDF’lerde her sayfada OCR’ı zorlar (daha yavaş).
      - ocrdpi=300    : OCR raster DPI (400-420 varsayılan; 300-360 hız için iyi denge).
      - lang=eng      : OCR dili.
      - pages=...     : PDF sayfa seçimi.
    """
    # default: json
    mode = "json"
    if "?" in req.url:
        qs = parse_qs(req.url.split("?", 1)[1])
        mode = (qs.get("mode", ["json"])[0] or "json").lower()

    # Hangi function'a delegasyon yapılacak?
    target_module_name = "extract_content" if mode != "raw" else "extract_content_raw"

    try:
        target_module = importlib.import_module(target_module_name)
    except Exception as e:
        return func.HttpResponse(
            f"Router error: failed to import '{target_module_name}': {e}",
            status_code=500
        )

    # Doğrudan ilgili function'ın main(req) çağrısı.
    # Aynı HttpRequest ile çağırıyoruz; böylece body yeniden kopyalanmıyor.
    try:
        resp = target_module.main(req)
    except Exception as e:
        return func.HttpResponse(
            f"Router error: failed to execute {target_module_name}.main: {e}",
            status_code=500
        )

    # Versiyon ve bilgi header'ı (opsiyonel ama faydalı)
    try:
        resp.headers["X-Extractor-Version"] = "1.3.0"
        resp.headers["X-Extractor-Mode"] = mode
    except Exception:
        pass

    return resp
