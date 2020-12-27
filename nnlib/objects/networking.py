from multiprocessing.connection import Listener
from multiprocessing.connection import Client
from .neural_network import *

def bind(address=('0.0.0.0', 16000)):
    from array import array
    with Listener(address) as listener:
        try:
            while True:
                with listener.accept() as client:
                    neural_network,x,y = client.recv()
                    deltas = neural_network.getDeltas(x,y)
                    client.send(deltas)
        except KeyboardInterrupt:
            pass

class NetworkedMLPNeuralNetwork(MLPNeuralNetwork):
    servers = [
        ('localhost',16000)
    ]

    def train(neural_network, training_set,verbose=False):
        try:
            error = neural_network.getError(training_set)
            _iter = 0
            servers = [Client(server) for server in neural_network.servers]
            for server in servers:
                server.send([neural_network,*random.choice(training_set)])
            while error > neural_network.MINIMUM_ERROR:
                for server in servers:
                    delta = server.recv()
                    neural_network.applyDeltas(delta)
                    server.close()
                error = neural_network.getError(training_set)
                servers = [Client(server) for server in neural_network.servers]
                for server in servers:
                    server.send([neural_network,*random.choice(training_set)])
                _iter += 1
                if (_iter % 10000) == 0 and verbose:
                    print(f"iter: {_iter}, error: {error}, servers: {len(servers)}")
        except KeyboardInterrupt:
            pass
        print(f"iter: {_iter}, error: {error}")

