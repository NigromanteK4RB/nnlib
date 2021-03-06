import concurrent.futures, os
from .neural_network import *

class ThreadedMLPNeuralNetwork(MLPNeuralNetwork):
    def train(neural_network, training_set,verbose=False):
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                error = neural_network.getError(training_set)
                _iter = 0
                threads = [executor.submit(neural_network.getDeltas,*random.choice(training_set)) for thread in range(os.cpu_count())]
                while error > neural_network.MINIMUM_ERROR:
                    concurrent.futures.wait(threads)
                    for thread in threads:
                        neural_network.applyDeltas(thread.result())
                    error = neural_network.getError(training_set)
                    threads = [executor.submit(neural_network.getDeltas,*random.choice(training_set)) for thread in range(os.cpu_count())]
                    _iter += 1
                    if (_iter % 10000) == 0 and verbose:
                        print(f"iter: {_iter}\nerror: {error}\nthreads: {len(threads)}\n")
        except KeyboardInterrupt:
            pass
        print(f"iter: {_iter}\nerror: {error}\nthreads: {len(threads)}\n")

