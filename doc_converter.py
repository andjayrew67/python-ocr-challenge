# doc_converter.py
"""
LibreOffice-based converter for .doc files to .docx format.
This module provides utilities to convert legacy Microsoft Word .doc files
to the modern .docx format using LibreOffice/OpenOffice.
"""

import os
import tempfile
import subprocess
import sys
from typing import Optional, Tuple

def _get_libreoffice_path() -> Optional[str]:
    """
    Find the LibreOffice installation path on Windows.
    """
    # Windows common install paths
    common_paths = [
        r"C:\Program Files\LibreOffice\program\soffice.exe",
        r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
        r"C:\Program Files\OpenOffice\program\soffice.exe",
        r"C:\Program Files (x86)\OpenOffice\program\soffice.exe",
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            return path
    
    # Try to find via Windows PATH
    try:
        result = subprocess.run(
            ["where", "soffice.exe"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            path = result.stdout.strip().split('\n')[0]
            if os.path.exists(path):
                return path
    except Exception:
        pass
    
    return None


def convert_doc_to_docx(doc_bytes: bytes, filename: str) -> Tuple[Optional[bytes], Optional[str]]:
    """
    Convert .doc file to .docx format using LibreOffice.
    
    Args:
        doc_bytes: Binary content of the .doc file
        filename: Original filename (for extension validation)
    
    Returns:
        Tuple[Optional[bytes], Optional[str]]: (converted docx bytes, error message if any)
        If successful: (docx_bytes, None)
        If failed: (None, error_message)
    """
    
    # Validate file extension
    ext = os.path.splitext(filename.lower())[1]
    if ext != ".doc":
        return None, f"Not a .doc file: {ext}"
    
    # Get LibreOffice path
    libreoffice_path = _get_libreoffice_path()
    if not libreoffice_path:
        return None, "LibreOffice/OpenOffice not found. Please install it."
    
    temp_dir = None
    try:
        # Create temporary directory for conversion
        temp_dir = tempfile.mkdtemp()
        
        # Write input .doc file to temp location
        input_file = os.path.join(temp_dir, "input.doc")
        with open(input_file, "wb") as f:
            f.write(doc_bytes)
        
        # Convert using LibreOffice headless mode
        # This converts .doc to .docx (Word XML format)
        convert_cmd = [
            libreoffice_path,
            "--headless",
            "--convert-to", "docx",
            "--outdir", temp_dir,
            input_file
        ]
        
        # Run conversion
        result = subprocess.run(
            convert_cmd,
            capture_output=True,
            text=True,
            timeout=60  # 60 second timeout
        )
        
        # Check for errors
        if result.returncode != 0:
            error_msg = result.stderr or result.stdout or "Conversion failed"
            return None, f"LibreOffice conversion error: {error_msg}"
        
        # Read converted .docx file
        output_file = os.path.join(temp_dir, "input.docx")
        if not os.path.exists(output_file):
            return None, "Conversion did not produce output file"
        
        with open(output_file, "rb") as f:
            docx_bytes = f.read()
        
        return docx_bytes, None
        
    except subprocess.TimeoutExpired:
        return None, "LibreOffice conversion timeout (>60 seconds)"
    except Exception as e:
        return None, f"Doc conversion error: {str(e)}"
    finally:
        # Cleanup temp directory
        if temp_dir and os.path.exists(temp_dir):
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except Exception:
                pass


def convert_doc_to_pdf_via_libreoffice(doc_bytes: bytes, filename: str) -> Tuple[Optional[bytes], Optional[str]]:
    """
    Alternative: Convert .doc file directly to PDF using LibreOffice.
    This is useful if you want to process PDFs instead of DOCX.
    
    Args:
        doc_bytes: Binary content of the .doc file
        filename: Original filename (for extension validation)
    
    Returns:
        Tuple[Optional[bytes], Optional[str]]: (converted pdf bytes, error message if any)
    """
    
    # Validate file extension
    ext = os.path.splitext(filename.lower())[1]
    if ext != ".doc":
        return None, f"Not a .doc file: {ext}"
    
    # Get LibreOffice path
    libreoffice_path = _get_libreoffice_path()
    if not libreoffice_path:
        return None, "LibreOffice/OpenOffice not found. Please install it."
    
    temp_dir = None
    try:
        # Create temporary directory for conversion
        temp_dir = tempfile.mkdtemp()
        
        # Write input .doc file to temp location
        input_file = os.path.join(temp_dir, "input.doc")
        with open(input_file, "wb") as f:
            f.write(doc_bytes)
        
        # Convert using LibreOffice headless mode to PDF
        convert_cmd = [
            libreoffice_path,
            "--headless",
            "--convert-to", "pdf",
            "--outdir", temp_dir,
            input_file
        ]
        
        # Run conversion
        result = subprocess.run(
            convert_cmd,
            capture_output=True,
            text=True,
            timeout=60  # 60 second timeout
        )
        
        # Check for errors
        if result.returncode != 0:
            error_msg = result.stderr or result.stdout or "Conversion failed"
            return None, f"LibreOffice conversion error: {error_msg}"
        
        # Read converted .pdf file
        output_file = os.path.join(temp_dir, "input.pdf")
        if not os.path.exists(output_file):
            return None, "Conversion did not produce output file"
        
        with open(output_file, "rb") as f:
            pdf_bytes = f.read()
        
        return pdf_bytes, None
        
    except subprocess.TimeoutExpired:
        return None, "LibreOffice conversion timeout (>60 seconds)"
    except Exception as e:
        return None, f"Doc conversion error: {str(e)}"
    finally:
        # Cleanup temp directory
        if temp_dir and os.path.exists(temp_dir):
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except Exception:
                pass


def is_doc_file(filename: str, content_type: Optional[str] = None) -> bool:
    """
    Check if a file is a legacy .doc file.
    
    Args:
        filename: The filename to check
        content_type: Optional MIME type
    
    Returns:
        bool: True if it's a .doc file
    """
    ext = os.path.splitext(filename.lower())[1]
    
    if ext == ".doc":
        return True
    
    # Check MIME type if provided
    if content_type:
        ctype_lower = content_type.lower()
        if "word" in ctype_lower and "processing" in ctype_lower:
            # application/msword or similar
            return ext == ".doc" or ext == ""
    
    return False
