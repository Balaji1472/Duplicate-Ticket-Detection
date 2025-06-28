from transformers import AutoTokenizer, AutoModel
import torch

class EmbeddingModel:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def embed_text(self, text: str) -> torch.Tensor:
        """Generate a sentence embedding using mean pooling"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]
        mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size())
        sum_embeddings = torch.sum(embeddings * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def embed_batch(self, texts: list) -> torch.Tensor:
        """Generate embeddings for a batch of texts"""
        return torch.stack([self.embed_text(text).squeeze() for text in texts])


#for testing
if __name__ == "__main__":
    texts = [
        "My internet is not working since morning.",
        "There is no internet connection at my home."
    ]
    model = EmbeddingModel()
    vectors = model.embed_batch(texts)
    print("Embedding shape:", vectors.shape)
