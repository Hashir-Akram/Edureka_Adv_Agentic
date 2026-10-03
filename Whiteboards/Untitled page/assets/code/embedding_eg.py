from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "What is Generative AI?",
    "Generative AI creates new content.",
    "The weather is hot today."
]

# Generate embeddings
embeddings = model.encode(sentences)

print("Number of sentences:", len(embeddings))
print("Embedding dimensions:", embeddings.shape)
print(embeddings)