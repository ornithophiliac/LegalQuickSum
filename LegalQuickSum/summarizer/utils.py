import pytesseract
from PIL import Image
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
import fitz  # PyMuPDF
import mimetypes

# Load the Pegasus Model
model_name = "google/pegasus-xsum"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)

def extract_text(file_path, content_type):
    """Extracts text from image or PDF files."""
    if content_type and content_type.startswith('image/'):
        try:
            img = Image.open(file_path)
            text = pytesseract.image_to_string(img)
            return text
        except Exception as e:
            print(f"Error processing image: {e}")
            return None

    elif content_type and content_type.startswith('application/pdf'):
        try:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            return text
        except Exception as e:
            print(f"Error processing PDF: {e}")
            return None

    elif content_type and content_type.startswith('text/'):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            return text
        except Exception as e:
            print(f"Error reading text file: {e}")
            return None

    else:
        print(f"Unsupported file type: {content_type}")
        return None

def summarize_text(text):
    """Summarize the extracted text using Pegasus, handling long texts."""
    MAX_LENGTH = 1024  # Adjust this based on your model's capabilities and average text length.
    try:
        if len(text) > MAX_LENGTH:
            # Handle long texts by chunking or other strategies.
            chunks = [text[i:i + MAX_LENGTH] for i in range(0, len(text), MAX_LENGTH)]
            summaries = []
            for chunk in chunks:
                tokens = tokenizer(chunk, truncation=True, padding="longest", return_tensors="pt")
                summary_ids = model.generate(**tokens)
                summaries.append(tokenizer.decode(summary_ids[0], skip_special_tokens=True))
            return " ".join(summaries)

        else:
            tokens = tokenizer(text, truncation=True, padding="longest", return_tensors="pt")
            summary_ids = model.generate(**tokens)
            return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    except Exception as e:
        print(f"Error during summarization with Pegasus: {e}")
        return "Error generating summary."


