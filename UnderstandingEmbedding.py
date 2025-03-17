import spacy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load spaCy's medium model (en_core_web_md)
nlp = spacy.load("en_core_web_lg")


# Function to get word vectors, with check for missing vectors
def get_vector(word):
    token = nlp(word)
    if token.has_vector:
        return token.vector
    else:
        return np.zeros(300)  # Return a zero vector if the word has no vector


# Vector arithmetic: Orange - Lemon + Papaya = ?
def vector_math(word1, word2, word3):
    vec1 = get_vector(word1)
    vec2 = get_vector(word2)
    vec3 = get_vector(word3)
    print(word1, word2, word3)
    print(vec1, vec2, vec3)
    return vec1 - vec2 + vec3


# Example Calculation: Orange - Lemon + Papaya
result_vector = vector_math("king", "man", "woman")


# Find the closest word in spaCy's vocab
def closest_word(vector):
    max_similarity = 1
    closest_word = ""
    for word in nlp.vocab:
        if word.has_vector:  # Only consider words with vectors
            similarity = np.dot(vector, word.vector)  # Cosine similarity
            if similarity > max_similarity:
                max_similarity = similarity
                closest_word = word.text
    return closest_word


predicted_word = closest_word(result_vector)
print("Predicted word (king - man + woman):", predicted_word)


# Visualization Function: Plot word vectors using PCA
def plot_embeddings(words):
    # Get word vectors for all words
    vectors = np.array([get_vector(word) for word in words])

    print("Vectors:", vectors)

    # Reduce dimensions with PCA to 2D for visualization
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(vectors)

    plt.figure(figsize=(16, 12))
    for i, word in enumerate(words):
        plt.scatter(reduced_vectors[i, 0], reduced_vectors[i, 1])
        plt.text(reduced_vectors[i, 0] + 0.02, reduced_vectors[i, 1] + 0.02, word, fontsize=12)

    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("Word Embeddings Visualization")
    plt.show()


# Example words to visualize
words_to_plot = ["king", "queen", "man", "woman", "doctor", "nurse", "paris", "france", "orange", "good", "great",
                 "better", "boy", "kid", "watermelon", "papaya", "fruit", "apple", "banana", "grape", "kiwi", "mango",]
plot_embeddings(words_to_plot)
