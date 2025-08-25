import os
import pytesseract
from PIL import Image
import pandas as pd
from datasets import load_dataset
import torch
from transformers import AutoProcessor, AutoModelForDocumentClassification
from diffusers import StableDiffusionPipeline
from google.colab import drive
import io

# Mount Google Drive for saving outputs
drive.mount('/content/drive')
OUTPUT_DIR = "/content/drive/MyDrive/MultiModalDocAI/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load dataset via API with error handling
try:
    dataset = load_dataset("rvl-cdip", split="train[:1000]")  # Stream 1000 samples
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# OCR function
def extract_text(image):
    try:
        text = pytesseract.image_to_string(image)
        return text if text.strip() else "No text extracted"
    except Exception as e:
        print(f"OCR error: {e}")
        return "OCR failed"

# Generate synthetic document images
def generate_synthetic_image(prompt="invoice document with text"):
    try:
        pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16)
        pipe = pipe.to("cuda")
        image = pipe(prompt).images[0]
        synthetic_path = os.path.join(OUTPUT_DIR, f"synthetic_{prompt.replace(' ', '_')}.png")
        image.save(synthetic_path)
        return synthetic_path, image
    except Exception as e:
        print(f"Synthetic image generation error: {e}")
        return None, None

# Preprocess streamed data
def preprocess_data():
    processed_data = []
    for idx, item in enumerate(dataset):
        try:
            image = Image.open(io.BytesIO(item["image"]))  # Stream image from memory
            label = item["label"]
            
            # Extract text
            text = extract_text(image)
            
            # Generate synthetic image
            synthetic_path, synthetic_image = generate_synthetic_image()
            synthetic_text = extract_text(synthetic_image) if synthetic_image else "No synthetic text"
            
            # Store in memory
            processed_data.append({
                "image_id": idx,
                "text": text,
                "label": label,
                "synthetic_image_path": synthetic_path if synthetic_path else "None",
                "synthetic_text": synthetic_text
            })
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue
    
    # Save to CSV in Google Drive
    if processed_data:
        df = pd.DataFrame(processed_data)
        df.to_csv(os.path.join(OUTPUT_DIR, "processed_data.csv"), index=False)
        print(f"Preprocessing complete. Data saved to {OUTPUT_DIR}")
    else:
        print("No data processed. Check dataset access or preprocessing steps.")

# Fine-tune DiT
def train_dit():
    try:
        processor = AutoProcessor.from_pretrained("microsoft/dit-base-finetuned-rvlcdip")
        model = AutoModelForDocumentClassification.from_pretrained("microsoft/dit-base-finetuned-rvlcdip", num_labels=16)
        model.to("cuda")
        
        # Load processed data
        df = pd.read_csv(os.path.join(OUTPUT_DIR, "processed_data.csv"))
        
        # Training loop
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        model.train()
        for idx, row in df.iterrows():
            try:
                image = Image.open(io.BytesIO(dataset[int(row["image_id"])]["image"]))  # Stream image
                encoding = processor(image, return_tensors="pt", truncation=True, padding="max_length")
                encoding = {k: v.to("cuda") for k, v in encoding.items()}
                labels = torch.tensor([row["label"]]).to("cuda")
                
                outputs = model(**encoding, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                if idx % 100 == 0:
                    print(f"Sample {idx}, Loss: {loss.item()}")
                    torch.cuda.empty_cache()  # Free memory
            except Exception as e:
                print(f"Training error at sample {idx}: {e}")
                continue
        
        # Save model to Google Drive
        model.save_pretrained(os.path.join(OUTPUT_DIR, "dit_model"))
        processor.save_pretrained(os.path.join(OUTPUT_DIR, "dit_processor"))
        print(f"Model saved to {OUTPUT_DIR}")
    except Exception as e:
        print(f"Training error: {e}")

if __name__ == "__main__":
    preprocess_data()
    train_dit()