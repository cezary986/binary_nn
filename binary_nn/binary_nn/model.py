from asyncio import as_completed
from typing import List, Tuple, Union
from logging import Logger, getLogger
from click import progressbar
from sklearn.metrics import balanced_accuracy_score
from .batch import Batcher
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np


class ActivationsCache:

    def __init__(self, n_layers: int, examples_count: int) -> None:
        self.activations: List[np.ndarray] = [
            [None for i in range(0, n_layers)] for j in range(0, examples_count)
        ]
        self.lock_index: int = None

    def lock_cache(self, layer_index: int):
        for cache in self.activations:
            for i in range(layer_index, len(cache)):
                cache[i] = None
        self.lock_index = layer_index

    def get_activation(self, layer_index: int, example_index: int) -> Union[np.ndarray, None]:
        return self.activations[example_index][layer_index]

    def cache_activation(self, layer_index: int, example_index: int, activation: np.ndarray) -> Union[np.ndarray, None]:
        if self.lock_index is not None and self.lock_index <= layer_index:
            return
        self.activations[example_index][layer_index] = activation


class BinaryNeuralNetwork:

    class LayerType:
        CONJUNCTION = True
        ALERNATIVE = False

    def __init__(
        self,
        hidden_layer_size: tuple,
        max_flips: int,
        random_state: int = None
    ) -> None:
        self._layers_size: List[int] = list(hidden_layer_size)
        self.coefs: List[np.ndarray] = None
        self.layers_types: List[BinaryNeuralNetwork.LayerType] = None
        self.inputs_count: int = None
        self.labels: np.ndarray = None
        self.outputs_count: int = None
        if random_state is not None:
            np.random.seed(random_state)
        self.max_flips: int = max_flips
        self.logger: Logger = getLogger('BinaryNN')
        self.cached_activations: ActivationsCache = None

    def _prepare_coefs(self, X: np.ndarray, y: np.ndarray):
        self.inputs_count = X.shape[1]
        self.labels = np.unique(y)
        self._layers_size.insert(0, self.inputs_count)
        self.coefs = []
        self.layers_types = []
        last_layer_type: BinaryNeuralNetwork.LayerType = BinaryNeuralNetwork.LayerType.ALERNATIVE
        inputs_count: int = self.inputs_count
        for i in range(1, len(self._layers_size)):
            neurons_count = self._layers_size[i]
            self.coefs.append(
                np.zeros((neurons_count, inputs_count), dtype=int))
            inputs_count = neurons_count
            # first layer is input layer
            if i > 0:
                layer_type = BinaryNeuralNetwork.LayerType.CONJUNCTION if \
                    last_layer_type == BinaryNeuralNetwork.LayerType.ALERNATIVE else \
                    BinaryNeuralNetwork.LayerType.ALERNATIVE
                self.layers_types.append(layer_type)
                last_layer_type = layer_type
            else:
                self.layers_types.append(None)
        self.cached_activations = ActivationsCache(
            n_layers=len(self.coefs), examples_count=X.shape[0])

    def _initialize_weights(self):
        for i in range(0, len(self.coefs)):
            self.coefs[i] = np.random.choice([0, 1], size=self.coefs[i].shape)
            # if i == 0:
            #     self.coefs[i] = np.ones(self.coefs[i].shape)
            if i == len(self.coefs) - 1:
                self.coefs[i] = np.ones(self.coefs[i].shape)

    def _flip_weight(self, coefs: List[np.ndarray], move: Tuple[int, int, int]):
        i, j, k = move
        coefs[i][j][k] = not coefs[i][j][k]

    def _optimize_weights(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, List[np.ndarray]]:
        best_coefs: List[np.ndarray] = [
            np.copy(weights) for weights in self.coefs
        ]
        best_acc: float = balanced_accuracy_score(y, self.predict(X))
        initial_acc: float = best_acc
        optimal: bool = False
        flip_count: int = 0
        while not optimal and flip_count < self.max_flips:
            best_move = None
            for i in range(len(self.coefs)):
                self.cached_activations.lock_cache(i)
                for j in range(0, best_coefs[i].shape[0]):
                    weights = best_coefs[i][j, :]
                    for k in range(0, weights.shape[0]):
                        weights[k] = not weights[k]
                        # tmp = self.coefs
                        # self.coefs = best_coefs
                        new_acc = balanced_accuracy_score(y, self.predict(X))
                        weights[k] = not weights[k]
                        # self.coefs = tmp
                        if new_acc > best_acc:
                            best_move = (i, j, k)
            if best_move is not None:
                self._flip_weight(best_coefs, best_move)
                flip_count += 1
            else:
                self.logger.debug(
                    'Finished optimizing weights: no better move possible')
                optimal = True
            if flip_count >= self.max_flips:
                self.logger.debug(
                    f'Finished optimizing weights: reached max_flips ({self.max_flips})')
        self.logger.debug(
            f'BAcc: {initial_acc} -> {best_acc} (n_flips = {flip_count})')
        return best_acc, best_coefs

    def fit(self, X: np.ndarray, y: np.ndarray, n_batches: int, n_epochs: int = 10):
        self._prepare_coefs(X, y)
        while True:
            self._initialize_weights()
            if balanced_accuracy_score(y, self.predict(X)) > 0:
                break
        batcher = Batcher(X, y)
        best_acc = 0
        best_coefs = self.coefs
        print(f'Start learning model n_epochs = {n_epochs}')
        for i in range(0, n_epochs):
            batches: List[Tuple[np.ndarray, np.ndarray]
                          ] = batcher.generate_batches(n_batches)
            self.coefs = [np.copy(e) for e in best_coefs]
            for j, batch in enumerate(batches):
                self.coefs = [np.copy(e) for e in best_coefs]
                X_b, y_b = batch
                print(
                    f' Epoch: {i + 1} batch: {j} --- initial BAcc = {best_acc} --- new BAcc = ...', end='\r')
                acc, coefs = self._optimize_weights(X_b, y_b)
                if acc > best_acc:
                    best_coefs = coefs
                    best_acc = acc
                    self.coefs = [np.copy(e) for e in coefs]
                print(
                    f' Epoch: {i + 1} batch: {j} --- initial BAcc = {best_acc} --- new BAcc = {best_acc}', end='\n')

        acc, coefs = self._optimize_weights(X, y)
        best_acc = acc
        self.coefs = coefs

    def predict(self, X: np.ndarray) -> np.ndarray:
        use_batches = False
        prediction = np.zeros(X.shape[0])
        if X.shape[0] > 100:
            use_batches = True
            batches = np.array_split(X, 8)

        def predict_batch(params):
            X = params[0]
            index = params[1]
            for example_index, example in enumerate(X):
                activation: np.ndarray = example
                for i, layer in enumerate(self.coefs):
                    layer_type: BinaryNeuralNetwork.LayerType = i % 2 == 0
                    cached_activation: np.ndarray = self.cached_activations.get_activation(
                        i, example_index)
                    if cached_activation is None:
                        next_activation: List[int] = []
                        for j, neuron in enumerate(layer):
                            weights: np.ndarray = self.coefs[i][j]
                            if layer_type == BinaryNeuralNetwork.LayerType.ALERNATIVE:
                                output = np.any(
                                    activation[weights == 1] == 1)
                            if layer_type == BinaryNeuralNetwork.LayerType.CONJUNCTION:
                                output = np.all(
                                    activation[weights == 1] == 1)
                            next_activation.append(output)
                        activation = np.array(next_activation, dtype=int)
                        self.cached_activations.cache_activation(
                            i, example_index, activation)
                    else:
                        activation = cached_activation
                prediction_index = example_index + index
                if np.any(activation == 1):
                    prediction[prediction_index] = 1
                else:
                    prediction[prediction_index] = 0
        if use_batches:
            with ThreadPoolExecutor() as executor:
                i = 0
                futures = []
                for batch in batches:
                    future = executor.submit(predict_batch, (batch, i))
                    futures.append(future)
                    i += batch.shape[0]
                for future in as_completed(futures):
                    pass
        else:
            predict_batch((X, 0))
        return prediction
