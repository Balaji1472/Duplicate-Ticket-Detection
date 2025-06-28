import spacy

# Load English model
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text: str) -> str:
    """
    Clean and normalize input text.
    - Lowercasing
    - Tokenization
    - Stopword removal
    - Lemmatization
    """
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

if __name__ == "__main__":
    sample = "I forgot my account password and need help resetting it."
    print("Original:", sample)
    print("Processed:", preprocess_text(sample))
