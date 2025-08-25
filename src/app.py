import streamlit as st
import pytesseract
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForDocumentClassification

# Configuration
MODEL_DIR = "./models"  # Adjust to local or Google Drive path
LABELS = [
    "letter", "form", "email", "handwritten", "advertisement", "scientific report",
    "scientific publication", "specification", "file folder", "news article",
    "budget", "invoice", "presentation", "questionnaire", "resume", "memo"
]

# Streamlit app
st.title("MultiModalDocAI: Document Classifier")

# Load model and processor
try:
    processor = AutoProcessor.from_pretrained(f"{MODEL_DIR}/dit_processor")
    model = AutoModelForDocumentClassification.from_pretrained(f"{MODEL_DIR}/dit_model")
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    st.write("Model loaded successfully.")
except Exception as e:
    st.write(f"Error loading model: {e}")
    st.stop()

# File uploader
uploaded_file = st.file_uploader("Upload a document image", type=["png", "jpg"])

if uploaded_file:
    # Load and display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Document", use_column_width=True)
    
    # Extract text
    try:
        text = pytesseract.image_to_string(image)
        st.write("### Extracted Text")
        st.write(text if text.strip() else "No text extracted")
    except Exception as e:
        st.write(f"OCR error: {e}")
    
    # Classify document
    try:
        encoding = processor(image, return_tensors="pt", truncation=True, padding="max_length")
        encoding = {k: v.to("cuda" if torch.cuda.is_available() else "cpu") for k, v in encoding.items()}
        with torch.no_grad():
            outputs = model(**encoding)
        predicted_label = LABELS[torch.argmax(outputs.logits).item()]
        st.write("### Document Type")
        st.write(predicted_label)
    except Exception as e:
        st.write(f"Classification error: {e}")

st.write("Upload a document to analyze!")