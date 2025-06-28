import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import load_tickets
from src.preprocessing import preprocess_text
from src.embedding import EmbeddingModel
from src.duplicate_detector import find_duplicates

# Load and preprocess tickets
tickets = load_tickets("data/sample_tickets.json")
texts = [preprocess_text(ticket["text"]) for ticket in tickets]
ids = [ticket["ticket_id"] for ticket in tickets]

print("Preprocessed texts:", texts)

# Generate embeddings
model = EmbeddingModel()
embeddings = model.embed_batch(texts)

print("Embeddings shape:", embeddings.shape)

# Detect duplicates
duplicates = find_duplicates(texts, ids, embeddings.numpy(), threshold=0.50)

print("Duplicates:", duplicates)

# Show results
for dup in duplicates:
    print(f"\n🔁 Duplicate Found: {dup['ticket_1']} and {dup['ticket_2']}")
    print(f"Similarity: {dup['similarity']}")
    print(f"{dup['ticket_1']}: {dup['text_1']}")
    print(f"{dup['ticket_2']}: {dup['text_2']}")
