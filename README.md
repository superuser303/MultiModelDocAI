# MultiModalDocAI

A multimodal document understanding system that classifies documents (e.g., invoices, letters) using the `microsoft/dit-base-finetuned-rvlcdip` model, extracts text with Tesseract OCR, and generates synthetic images with Stable Diffusion. Built for my GitHub portfolio: [superuser303](https://github.com/superuser303).

## Features
- Streams RVL-CDIP dataset from Hugging Face.
- Extracts text using Tesseract OCR.
- Generates synthetic document images with Stable Diffusion.
- Fine-tunes a Document Image Transformer (DiT) for classification.
- Streamlit app for real-time document analysis.

## Setup
1. Clone the repo:
   ```bash
   git clone https://github.com/superuser303/MultiModalDocAI.git
   cd MultiModalDocAI