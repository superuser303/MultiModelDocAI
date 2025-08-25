from PIL import Image
import pytesseract
import io

def extract_text(image):
    """Extract text from an image using Tesseract OCR."""
    try:
        text = pytesseract.image_to_string(image)
        return text if text.strip() else "No text extracted"
    except Exception as e:
        print(f"OCR error: {e}")
        return "OCR failed"

def load_image_from_bytes(image_bytes):
    """Load an image from bytes."""
    try:
        return Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        print(f"Image loading error: {e}")
        return None