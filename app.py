import re
import random
from itertools import product
import pandas as pd

from libs.training import *
from libs.evaulation import *


def grid_search():
    global word2idx
    df_read = pd.read_csv("news_articles.csv")
    news_data = df_read["description"].dropna().tolist()

    sample_text = [re.findall(r'\b\w+\b', desc.lower()) for desc in news_data]
    test_words = list(set(word for desc in sample_text[:10] for word in desc))
    # print("\nsample-text: ", sample_text)
    # print("\ntest-words: ", test_words)

    epochx = 100
    window_sizes = [1, 2, 3]
    embedding_dims = [50, 100, 200]

    best_model = None
    best_loss = float('inf')
    best_params = None

    for window_size, embedding_dim in product(window_sizes, embedding_dims):
        print(f"Training with window_size={window_size}, embedding_dim={embedding_dim}")
        training_pairs, word2idx = generate_training_data(sample_text, window_size=window_size)
        # print("\ntraining-pairs: ", training_pairs)
        # print("\nword2idx: ", word2idx)

        model, avg_loss = train_model(training_pairs, word2idx, vocab_size=len(word2idx),
                                      embedding_dim=embedding_dim, epochs=epochx)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model = model
            best_params = (window_size, embedding_dim)

    print(f"\n\nBest Model: window_size={best_params[0]}, embedding_dim={best_params[1]}, AVG-Loss={best_loss}")

    # Similarity
    sample_word = random.choice(test_words)
    similar_words = word_similarity(sample_word, word2idx, best_model.W1)
    print(f"Words similar to '{sample_word}': {similar_words}")


if __name__ == "__main__":
    grid_search()
