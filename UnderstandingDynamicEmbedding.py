from sentence_transformers import SentenceTransformer, util

# Load a pretrained BERT-based model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define two sentences for similarity comparison
sentence1 = "The quick brown fox jumps over the lazy dog."
sentence2 = "A fast dark-colored fox leaps across a sleepy canine."
sentence3 = "I deposited money in the bank."
sentence4 = "He sat by the river bank."

# Generate dynamic embeddings
embedding1 = model.encode(sentence1, convert_to_tensor=True)
embedding2 = model.encode(sentence2, convert_to_tensor=True)
embedding3 = model.encode(sentence3, convert_to_tensor=True)
embedding4 = model.encode(sentence4, convert_to_tensor=True)

# Compute cosine similarity
cosine_sim = util.pytorch_cos_sim(embedding1, embedding2)
cosine_sim2 = util.pytorch_cos_sim(embedding3, embedding4)

print(f"Cosine Similarity of First two sentences: {cosine_sim.item():.4f}")
print(f"Cosine Similarity of Last two sentences: {cosine_sim2.item():.4f}")
