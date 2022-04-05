from typing import Callable, List, Tuple
import numpy as np
import random
from logging import Logger, getLogger
from sklearn.metrics import balanced_accuracy_score

from binary_nn.binary_nn.batch import Batcher


class Model:

    ALTERNATIVE = 'alt'
    CONJUNCTION = 'conj'

    def __init__(
        self,
        average_rule_length: float,
        initialization_probability: float,
        hidden_layer_size: tuple,
        max_flips: int = 10,
        random_state: int = None
    ) -> None:
        self.l: float = average_rule_length
        self.p: float = initialization_probability
        self.hidden_layer_size: tuple = hidden_layer_size
        if random_state is not None:
            np.random.seed(random_state)
            random.seed = random_state
        self.logger: Logger = getLogger('BinaryNN')
        self.coefs: List[np.ndarray] = []
        self.layers_types: List[str] = []
        self.max_flips: int = max_flips

    def initialize_weights(self, X: np.ndarray):
        # attributes + their negations
        attributes_count: int = int(X.shape[1] / 2)

        # first layer
        neurons_count: int = self.hidden_layer_size[0]
        inputs_count: int = X.shape[1]
        weights: np.ndarray = np.zeros(
            (neurons_count, inputs_count), dtype=int)
        probability = self.l / attributes_count
        for neuron_index in range(0, neurons_count):
            selected_attributes: np.ndarray = np.random.binomial(
                n=1, p=probability, size=[int(attributes_count)]).astype(bool)
            attr_index: int = 0
            for _ in selected_attributes[selected_attributes == True]:
                # select attr or negation with prob 0.5
                if random.uniform(0, 1) >= 0.5:
                    weights[neuron_index, attr_index] = 1
                else:
                    weights[neuron_index, attr_index] = 0
                attr_index += 2
        self.coefs.append(weights)
        self.layers_types.append(Model.CONJUNCTION)

        # other layers
        for layer_index, neurons_count in enumerate(self.hidden_layer_size[1:]):
            inputs_count: int = self.hidden_layer_size[layer_index]
            weights: np.ndarray = np.random.binomial(
                n=1, p=self.p, size=(neurons_count, inputs_count))
            self.coefs.append(weights)
            if self.layers_types[-1] == Model.CONJUNCTION:
                self.layers_types.append(Model.ALTERNATIVE)
            else:
                self.layers_types.append(Model.CONJUNCTION)
            # check if at least one outgoing neuron connection is connected
            for input_index in range(0, inputs_count):
                if np.all(weights[:, input_index] == 0):
                    # flip random connection to 1
                    weights[random.randint(
                        0, neurons_count - 1), input_index] = 1

        # last output layer
        neurons_count: int = 1
        inputs_count: int = self.hidden_layer_size[-1]
        weights: np.ndarray = np.ones((neurons_count, inputs_count), dtype=int)
        self.coefs.append(weights)
        # last layer is always alternative
        self.layers_types.append(Model.ALTERNATIVE)

    def _flip_weight(self, coefs: List[np.ndarray], move: Tuple[int, int, int]):
        i, j, k = move
        coefs[i][j, k] = not coefs[i][j, k]

    def _optimize_weights(
            self,
            X: np.ndarray,
            y: np.ndarray,
            best_quality: float,
            train_measure: Callable[[np.ndarray, np.ndarray], float]) -> Tuple[float, List[np.ndarray]]:
        coefs: List[np.ndarray] = [np.copy(weights) for weights in self.coefs]
        initial_quality: float = best_quality
        best_move: Tuple[int, int, int] = None
        optimal: bool = False
        flip_count: int = 0
        while not optimal and flip_count < self.max_flips:
            best_move = None
            for layer_index, layer_weights in enumerate(coefs):
                neurons_count: int = layer_weights.shape[0]
                inputs_count: int = layer_weights.shape[1]
                for neuron_index in range(0, neurons_count):
                    for input_index in range(0, inputs_count):
                        move = (layer_index, neuron_index, input_index)
                        self._flip_weight(coefs, move)
                        quality: float = train_measure(y, self.predict(X))
                        if quality > best_quality:
                            best_quality = quality
                            best_move = move
                        # flip again to recover original structure
                        self._flip_weight(coefs, move)
            if best_move is not None:
                self._flip_weight(coefs, best_move)
                flip_count += 1
            else:
                self.logger.debug(
                    'Finished optimizing weights: no better move possible')
                optimal = True
            if flip_count >= self.max_flips:
                self.logger.debug(
                    f'Finished optimizing weights: reached max_flips ({self.max_flips})')
        if best_quality > initial_quality:
            self.logger.debug(
                f'Increased quality from: {initial_quality} -> {best_quality}  n_flips={flip_count}')
        return best_quality, coefs

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_batches: int,
        n_epochs: int = 5,
        train_measure: Callable[[np.ndarray, np.ndarray],
                                float] = balanced_accuracy_score
    ):
        self.initialize_weights(X)
        batcher: Batcher = Batcher(X, y)
        best_quality: float = train_measure(y, self.predict(X))
        best_coefs: np.ndarray = self.coefs
        batches: List[Tuple[np.ndarray, np.ndarray]] = None
        print(
            f'Start learning model n_epochs = {n_epochs}, initial quality = {best_quality}')
        try:
            for i in range(0, n_epochs):
                self.coefs = best_coefs
                print(f'Epoch: {i}')
                batches = batcher.generate_batches(n_batches)
                for j, batch in enumerate(batches):
                    X_b, y_b = batch
                    quality, coefs = self._optimize_weights(
                        X_b, y_b, best_quality, train_measure)
                    if quality > best_quality:
                        self.logger.debug(
                            f'New best quality found {quality} was {best_quality}')
                        best_coefs = coefs
                        best_quality = quality
                        
        except KeyboardInterrupt:
            print('Interupt learning, skipping epochs...')

        self.coefs = best_coefs
        # optimize last time on full dataset
        acc, coefs = self._optimize_weights(X, y, best_quality, train_measure)
        self.logger.debug(
            f'Final model quality found {train_measure(y, self.predict(X)) } was {best_quality}')
        self.coefs = coefs

    def predict(self, X: np.ndarray) -> np.ndarray:
        prediction: np.ndarray = np.zeros(X.shape[0], dtype=int)
        for i, example in enumerate(X):
            inputs: np.ndarray = example
            activation: np.ndarray = None
            for layer_weights in self.coefs:
                activation: np.ndarray = np.dot(
                    np.logical_not(inputs), layer_weights.T)
                inputs = activation.astype(bool)
            if np.any(activation == 1):
                prediction[i] = 1
        return prediction
