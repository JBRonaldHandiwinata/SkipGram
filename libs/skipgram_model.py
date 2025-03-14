import numpy as np


class SkipGramModel:
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.W1 = np.random.randn(vocab_size, embedding_dim)
        self.W2 = np.random.randn(embedding_dim, vocab_size)

    def forward(self, one_hot_vector):
        hidden_layer = np.dot(one_hot_vector, self.W1)
        output_layer = np.dot(hidden_layer, self.W2)
        output_layer = self._softmax(output_layer)
        return hidden_layer, output_layer

    def backward(self, one_hot_vector, target_vector, learning_rate=0.01):
        hidden_layer, output_layer = self.forward(one_hot_vector)
        # error = target_vector - output_layer
        error = output_layer - target_vector

        # Compute Gradients
        output_layer_gradient = np.outer(hidden_layer, error)
        hidden_layer_gradient = np.outer(one_hot_vector, np.dot(self.W2, error))

        # Update Weights
        self.W1 -= learning_rate * hidden_layer_gradient
        self.W2 -= learning_rate * output_layer_gradient

    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
