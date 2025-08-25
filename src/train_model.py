import os
import pandas as pd
from PIL import Image
from datasets import load_dataset
import torch
from transformers import AutoProcessor, AutoModelForDocumentClassification
import io

# Configuration
OUTPUT_DIR = "/content/drive/MyDrive/MultiModalDocAI/processed"
MODEL_DIR = "/content/drive/MyDrive/MultiModalDocAI/models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Fine-tune DiT
def train_dit():
    try:
        processor = AutoProcessor.from_pretrained("microsoft/dit-base-finetuned-rvlcdip")
        model = AutoModelForDocumentClassification.from_pretrained("microsoft/dit-base-finetuned-rvlcdip", num_labels=16)
        model.to("cuda")
        
        # Load dataset for image streaming
        dataset = load_dataset("rvl-cdip", split="train[:1000]")
        print("Dataset loaded successfully.")
        
        # Load processed data
        df = pd.read_csv(os.path.join(OUTPUT_DIR, "processed_data.csv"))
        
        # Training loop
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        model.train()
        for idx, row in df.iterrows():
            try:
                image = Image.open(io.BytesIO(dataset[int(row["image_id"])]["image"]))
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
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"Training error at sample {idx}: {e}")
                continue
        
        # Save model
        model.save_pretrained(os.path.join(MODEL_DIR, "dit_model"))
        processor.save_pretrained(os.path.join(MODEL_DIR, "dit_processor"))
        print(f"Model saved to {MODEL_DIR}")
    except Exception as e:
        print(f"Training error: {e}")

if __name__ == "__main__":
    from google.colab import drive
    drive.mount('/content/drive')
    train_dit()