
from .objects.neuron import MLPNeuron as Neuron
from .objects.neural_layer import MLPNeuralLayer as NeuralLayer
from .objects.neural_network import MLPNeuralNetwork as NeuralNetwork


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
    print(neural_network(_set[0]))
