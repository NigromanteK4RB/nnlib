#!/usr/bin/env python3.9
from nnlib import MLPNeuron as Neuron
from nnlib import MLPNeuralLayer as NeuralLayer
from nnlib import ThreadedMLPNeuralNetwork as NeuralNetwork


neural_network = NeuralNetwork([
    NeuralLayer([Neuron(),Neuron(),]),
    NeuralLayer([Neuron(),]),
])

training_set = [
    [[0,0],[0]],
    [[0,1],[1]],
    [[1,0],[1]],
    [[1,1],[0]],
]

neural_network.train(training_set,True)

for _set in training_set:
    _nout = neural_network(_set[0])[0]
    _nout = round(_nout,3)
    _tout = _set[1][0]
    _tout = round(_tout,3)
    error = _nout - _tout
    error = round(error,3)
    print(f'Y: {_tout}\ny: {_nout}\nE: {error}\n')

