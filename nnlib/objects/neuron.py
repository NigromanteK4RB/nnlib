import math
import random

class BaseNeuron(object):
    bias = float()
    weights = list()

    def computeOutput(neuron,inputs):
        raise NotImplementedError

    __call__ = computeOutput

    def sum(neuron):
        raise NotImplementedError

    @staticmethod
    def activation(input):
        raise NotImplementedError

    @staticmethod
    def activationDerivative(input):
        raise NotImplementedError

    @staticmethod
    def cost(net_output,target_output):
        raise NotImplementedError

    @staticmethod
    def costDerivative(net_output,target_output):
        raise NotImplementedError

class MLPNeuron(BaseNeuron):

    def computeOutput(neuron,inputs):
        if len(neuron.weights) != (inputs_len := len(inputs)):
            neuron.weights = [random.random() for i in range(inputs_len)]
            neuron.bias = random.random()
        neuron.inputs = inputs
        neuron.output = neuron.activation(neuron.sum())
        return neuron.output

    __call__ = computeOutput

    def sum(neuron):
        return sum([x*y for x,y in zip(neuron.inputs,neuron.weights)]) + neuron.bias
    
    @staticmethod
    def activation(input):
        return 1 / (1 + math.exp(-input))

    @staticmethod
    def activationDerivative(input):
        return input * (1 - input)

    @staticmethod
    def cost(net_output,target_output):
        return 0.5 * (target_output - net_output) ** 2

    @staticmethod
    def costDerivative(net_output,target_output):
        return -(target_output - net_output)

