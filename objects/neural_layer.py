
class BaseNeuralLayer(list):
    def sum(neural_layer, inputs):
        raise NotImplementedError

    __call__ = sum

    def getOutputs(neural_layer):
        raise NotImplementedError

class MLPNeuralLayer(BaseNeuralLayer):

    def sum(neural_layer, inputs):
        return [neuron(inputs) for neuron in neural_layer]
    
    def getOutputs(neural_layer):
        return [neuron.output for neuron in neural_layer]

