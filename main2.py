from tkinter import N
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from _ import *
import logging
import numpy as np
from binary_nn.binary_nn.model2 import Model
from datasets.furnkranz import get_dataset

X, y = get_dataset(500)

ones_count: int = y[y == 1].shape[0]
zeros_count: int = y[y == 1].shape[0]

if zeros_count > ones_count:
    print('Flip classes')
    y = np.logical_not(y).astype(int)

logging.basicConfig(level=logging.DEBUG)
disable_sklearn_warnings()

# import numpy as np
# y = np.logical_not(y).astype(int)

model = Model(
    hidden_layer_size=(32, 16, 8, 4, 2),
    average_rule_length=2,
    initialization_probability=0.1,
    max_flips=10
)

model.fit(X, y, n_batches=10, n_epochs=100)

prediction = model.predict(X)


print(prediction.astype(int))
print(y)

BAcc = accuracy_score(y, prediction)

print(f'BAcc = {BAcc}')
