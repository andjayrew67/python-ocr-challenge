# Doc Extractor — README

This repository provides a robust document extraction service (PDF-first) with a CLI and Azure Functions HTTP endpoints.
The project supports native text extraction using PyMuPDF and optional OCR fallbacks (Tesseract / ocrmypdf). It also contains helpers and HTTP functions to run locally.

Below are focused instructions for running this project on Windows (PowerShell) and general notes for macOS/Linux.

---

## Minimum requirements

- Python 3.10+
- (Optional but recommended) Tesseract OCR installed for image/OCR handling
- (Optional) ocrmypdf system tools when you want OCR pre-pass (requires `ghostscript`, `qpdf` on some platforms)
- Azure Functions Core Tools (to run the functions locally if you want to test the Azure function endpoints)

Note: The project `requirements.txt` contains the Python dependencies used by the functions and CLI.

---

## Quick setup (Windows PowerShell)

1. Create and activate a virtual environment (recommended):

```powershell
python -m venv .venv
# PowerShell activation
.\.venv\Scripts\Activate.ps1
```

If the execution policy blocks activation in PowerShell, run this temporarily (admin not required):

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\.venv\Scripts\Activate.ps1
```

2. Install Python dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Notes:
- `requirements.txt` includes core packages: azure-functions, PyMuPDF, Pillow, pytesseract, pdfplumber/camelot (optional), ocrmypdf (optional), OpenCV, and more.

---

## Install external/binary dependencies

- Tesseract (Windows):
  - Download from https://github.com/UB-Mannheim/tesseract/wiki (recommended Windows builds) or https://github.com/tesseract-ocr/tesseract
  - Install and ensure path is `C:\Program Files\Tesseract-OCR\tesseract.exe` (the code already checks and sets this path if present).

- ocrmypdf optional pre-pass: required system deps on many OSs: qpdf, ghostscript. See: https://ocrmypdf.readthedocs.io/

---

## Azure Functions (local)

This project contains multiple Azure Functions:
- `extract_content` -> POST /api/extract/json (JSON output)
- `extract_content_raw` -> POST /api/extract/raw (RAW text output)
- `router` -> POST /api/router?mode=json|raw (delegates to above)

Start the local host (you need Azure Functions Core Tools installed):

```powershell
# from repository root
func host start
```

VS Code users can also use the pre-defined workspace task `func: host start` in the Run/Tasks palette.

By default the functions use `local.settings.json` values (AzureWebJobsStorage=UseDevelopmentStorage=true). If you need storage, install Azurite or configure storage connection strings.

### Example: call the function (PowerShell - using curl or Invoke-WebRequest)

# Using curl.exe (PowerShell ships a curl alias — use curl.exe to call the actual binary):
```powershell
curl.exe -v -X POST "http://localhost:7071/api/extract/json" -F "files=@C:\path\to\sample.pdf" 
```

# Using PowerShell's Invoke-RestMethod (multipart requires more work) — simplest: use curl.exe or Postman for multipart file uploads.

Example using `router` and query flags:

```powershell
curl.exe -v -X POST "http://localhost:7071/api/router?mode=json&fast=1&lang=eng" -F "files=@C:\path\to\mydoc.pdf"
```

The functions accept many query params (see function sources):
- forceocr, fast, ocrdpi, lang, pages, workers, preocr, tables (camelot|plumber), tables_out (csv|json|both), images

---

## CLI (pdf_extractor.py)

You can run the CLI locally against a PDF file. Examples:

```powershell
# JSON output (default)
python pdf_extractor.py C:\path\to\document.pdf --format json

# RAW text output
python pdf_extractor.py C:\path\to\document.pdf --format raw

# Force OCR on all pages
python pdf_extractor.py C:\path\to\document.pdf --force-ocr --no-ocrmypdf

# Extract only pages 2-5
python pdf_extractor.py C:\path\to\document.pdf --page-range 2-5

# Higher DPI for OCR
python pdf_extractor.py C:\path\to\document.pdf --dpi 400
```

The CLI supports many flags — check the source header for more options.

---

## Example: run functions with a simple curl (working sample)

1. Start functions host (see section above)
2. Call endpoint using curl.exe:

```powershell
curl.exe -v -X POST "http://localhost:7071/api/extract/json" -F "files=@C:\path\to\sample.pdf"
```

The function will return a JSON payload describing pages and extracted content.

---

## Troubleshooting

- If Tesseract is not found or OCR doesn't work, ensure Tesseract is installed and on PATH or installed in the default Windows location (C:\Program Files\Tesseract-OCR\tesseract.exe).
- On Windows PowerShell, if virtualenv activation fails, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` before activation.
- If `func host start` complains about missing Azure Functions Core Tools, install it: https://learn.microsoft.com/en-us/azure/azure-functions/functions-run-local
- If `ocrmypdf` errors occur, verify `ghostscript`/`qpdf` are installed and available on PATH.

---

## Files to know

- `pdf_extractor.py` — CLI & core extraction logic
- `extract_file_content.py` — wrapper for file-bytes extraction
- `extract_content` (folder) — Azure Function returning JSON
- `extract_content_raw` (folder) — Azure Function returning RAW text
- `router` — lightweight router that delegates to json/raw extractors
- `requirements.txt` — Python dependencies (core + optional)

---

If you'd like, I can add a small example script to test the endpoints or a sample YAML snipped for Docker-based attempts. Want me to add a sample test curl script or quick unit test harness next?
