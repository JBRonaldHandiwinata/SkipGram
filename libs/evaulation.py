from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans


def word_similarity(word, word2idx, embedding_matrix, top_n=5):
    if word not in word2idx:
        print(f"Word '{word}' not in vocabulary.")
        return []
    word_vector = embedding_matrix[word2idx[word]].reshape(1, -1)
    similarities = cosine_similarity(word_vector, embedding_matrix)[0]
    similar_words = sorted(word2idx.keys(), key=lambda w: similarities[word2idx[w]], reverse=True)[1:top_n+1]
    return similar_words


def semantic_clustering(word2idx, word_vectors, num_clusters=5):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(word_vectors)

    clustered_words = {i: [] for i in range(num_clusters)}
    for word, cluster in zip(word2idx.keys(), clusters):
        clustered_words[cluster].append(word)
    return clustered_words
