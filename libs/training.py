import numpy as np
from .skipgram_model import SkipGramModel


def generate_training_data(words, window_size=2):
    all_words = [word for sentence in words for word in sentence]
    word2idx = {word: idx for idx, word in enumerate(set(all_words))}
    training_pairs = []
    for sentence in words:
        for i, target_word in enumerate(sentence):
            context_range = range(max(0, i - window_size), min(len(sentence), i + window_size + 1))
            for j in context_range:
                if i != j:
                    training_pairs.append((target_word, sentence[j]))
    return training_pairs, word2idx


def train_model(training_pairs, word2idx, vocab_size, embedding_dim=100, epochs=10):
    model = SkipGramModel(vocab_size, embedding_dim)
    loss_history = []
    for epoch in range(epochs):
        total_loss = 0
        for target_word, context_word in training_pairs:
            target_vector = np.zeros(vocab_size)
            target_vector[word2idx[target_word]] = 1

            context_vector = np.zeros(vocab_size)
            context_vector[word2idx[context_word]] = 1

            hidden, output = model.forward(target_vector)
            loss = -np.log(output[word2idx[context_word]])

            model.backward(target_vector, context_vector)
            total_loss += loss

        avg_loss = total_loss / len(training_pairs)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss}")
    return model, np.mean(loss_history)
