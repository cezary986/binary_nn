from tkinter import N
from sklearn.metrics import balanced_accuracy_score
from _ import *
import logging
from binary_nn.binary_nn.model import BinaryNeuralNetwork
from datasets.furnkranz import get_dataset

X, y = get_dataset(100)


logging.basicConfig(level=logging.DEBUG)
disable_sklearn_warnings()

model = BinaryNeuralNetwork(
    hidden_layer_size=(32, 16),
    max_flips=10,
    # random_state=1
)

model.fit(X, y, n_batches=3, n_epochs=5)

prediction = model.predict(X)

print(prediction)
print(y)

BAcc = balanced_accuracy_score(y, prediction)

print(f'BAcc = {BAcc}')
