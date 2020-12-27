import random

class BaseNeuralNetwork(list):
    ALPHA = 0.4
    MINIMUM_ERROR = 0.0000009

    def sum(neural_network,inputs):
        raise NotImplementedError

    __call__ = sum

    def getDeltas(neural_network, inputs, outputs):
        raise NotImplementedError

    def applyDeltas(neural_network, deltas):
        raise NotImplementedError

    def getError(neural_network, training_set):
        raise NotImplementedError

    def train(neural_network,training_set,verbose=False):
        raise NotImplementedError

class MLPNeuralNetwork(BaseNeuralNetwork):

    def sum(neural_network, inputs):
        outputs = inputs
        for neural_layer in neural_network:
            outputs = neural_layer(outputs)
        return outputs

    __call__ = sum

    def getDeltas(neural_network, inputs, outputs):
        neural_network(inputs)

        net_weights_deltas = []
        net_biases_deltas = []

        for neural_layer in reversed(neural_network):
            layer_weights_deltas = []
            layer_biases_deltas = []
            neuron_weights_derivatives = []
            if neural_layer == neural_network[-1]:
                for n,neuron in enumerate(neural_layer):
                    neuron_weights_deltas = []
                    neuron_weight_derivative = neuron.costDerivative(neuron.output,outputs[n]) * neuron.activationDerivative(neuron.output)
                    neuron_bias_delta = neuron_weight_derivative
                    for weight in range(len(neuron.weights)):
                        weight_delta = neuron.inputs[weight] * neuron_weight_derivative
                        neuron_weights_deltas.append(weight_delta)
                    layer_weights_deltas.append(neuron_weights_deltas)
                    layer_biases_deltas.append(neuron_bias_delta)
                    neuron_weights_derivatives.append(neuron_weight_derivative)
            else:
                for n,neuron in enumerate(neural_layer):
                    neuron_weights_deltas = []
                    error = 0
                    for n,derivative in enumerate(_neuron_weights_derivatives):
                        error += derivative * _neural_layer[n].weights[n]
                    neuron_weights_derivative = error * neuron.activationDerivative(neuron.output)
                    neuron_bias_delta = neuron_weights_derivative
                    for weight in range(len(neuron.weights)):
                        weight_delta = neuron_weights_derivative * neuron.inputs[weight]
                        neuron_weights_deltas.append(weight_delta)
                    layer_weights_deltas.append(neuron_weights_deltas)
                    layer_biases_deltas.append(neuron_bias_delta)
                    neuron_weights_derivatives.append(neuron_weights_derivative)
            _neuron_weights_derivatives = neuron_weights_derivatives
            _neural_layer = neural_layer
            net_weights_deltas.insert(0,layer_weights_deltas)
            net_biases_deltas.insert(0,layer_biases_deltas)
        return net_weights_deltas,net_biases_deltas

    def applyDeltas(neural_network,deltas):

        for neural_layer in range(len(neural_network)):
            for neuron in range(len(neural_network[neural_layer])):
                for weight in range(len(neural_network[neural_layer][neuron].weights)):
                    delta = deltas[0][neural_layer][neuron][weight]
                    neural_network[neural_layer][neuron].weights[weight] -= neural_network.ALPHA * delta
                delta = deltas[1][neural_layer][neuron]
                neural_network[neural_layer][neuron].bias -= neural_network.ALPHA * delta

    def getError(neural_network,training_set):
        total_error = 0
        for _set in range(len(training_set)):
            x, y = training_set[_set]
            neural_network(x)
            for output in range(len(y)):
                neuron = neural_network[-1][output]
                total_error += neuron.cost(neuron.output,y[output])
        return total_error

    def train(neural_network,training_set,verbose=False):
        try:
            error = neural_network.getError(training_set)
            _iter = 0
            while error > neural_network.MINIMUM_ERROR:
                x,y = random.choice(training_set)
                deltas = neural_network.getDeltas(x,y)
                neural_network.applyDeltas(deltas)
                error = neural_network.getError(training_set)
                _iter += 1
                if (_iter % 10000) == 0 and verbose:
                    print(_iter,error)
        except KeyboardInterrupt:
            pass
        print(_iter,error)

