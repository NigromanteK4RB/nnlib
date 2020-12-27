import math
import random

class Neurona:
    sesgo = float()
    pesos = list()

    def calcularSalidas(neurona, entradas):
        if len(neurona.pesos) != (entradas_len := len(entradas)):
            neurona.pesos = [random.random() for i in range(entradas_len)]
            neurona.sesgo = random.random()
        neurona.entradas = entradas
        neurona.salida = neurona.funcionActivacion(neurona.sumar())
        return neurona.salida

    __call__ = calcularSalidas

    def sumar(neurona):
        return sum([x*y for x,y in zip(neurona.entradas,neurona.pesos)]) + neurona.sesgo
    
    @staticmethod
    def funcionActivacion(entrada):
        return 1 / (1 + math.exp(-entrada))

    @staticmethod
    def funcionActivacionDerivada(entrada):
        return entrada * (1 - entrada)

    @staticmethod
    def funcionCoste(salida_obtenida, salida_deseada):
        return 0.5 * (salida_deseada - salida_obtenida) ** 2

    @staticmethod
    def funcionCosteDerivada(salida_obtenida, salida_deseada):
        return -(salida_deseada - salida_obtenida)

class CapaNeuronal(list):

    def sumar(capa_neuronal, entradas):
        return [neurona(entradas) for neurona in capa_neuronal]

    __call__ = sumar
    
    def obtener_salidas(capa_neuronal):
        return [neurona.salida for neurona in capa_neuronal]

class RedNeuronal(list):
    ALFA = 0.5
    ERROR_MINIMO = 0.0001

    def sumar(red_neuronal, entradas):
        salida = entradas
        for capa_neuronal in red_neuronal:
            salida = capa_neuronal(salida)
        return salida

    __call__ = sumar

    def obtenerDeltas(red_neuronal, entradas, salidas):
        red_neuronal(entradas)

        deltas_pesos_red = []
        deltas_sesgos_red = []

        for capa_neuronal in reversed(red_neuronal):
            deltas_pesos_capa = []
            deltas_sesgos_capa = []
            derivadas_pesos_neuronas = []
            if capa_neuronal == red_neuronal[-1]:
                for numero,neurona in enumerate(capa_neuronal):
                    deltas_pesos_neurona = []
                    derivada_peso_neurona = neurona.funcionCosteDerivada(neurona.salida,salidas[numero]) * neurona.funcionActivacionDerivada(neurona.salida)
                    delta_sesgo_neurona = derivada_peso_neurona
                    for peso in range(len(neurona.pesos)):
                        delta_peso = neurona.entradas[peso] * derivada_peso_neurona
                        deltas_pesos_neurona.append(delta_peso)
                    deltas_pesos_capa.append(deltas_pesos_neurona)
                    deltas_sesgos_capa.append(delta_sesgo_neurona)
                    derivadas_pesos_neuronas.append(derivada_peso_neurona)
            else:
                for numero,neurona in enumerate(capa_neuronal):
                    deltas_pesos_neurona = []
                    error = 0
                    for n,derivada in enumerate(_derivadas_pesos_neuronas):
                        error += derivada * _capa_neuronal[n].pesos[numero]
                    derivada_peso_neurona = error * neurona.funcionActivacionDerivada(neurona.salida)
                    delta_sesgo_neurona = derivada_peso_neurona
                    for peso in range(len(neurona.pesos)):
                        delta_peso = derivada_peso_neurona * neurona.entradas[peso]
                        deltas_pesos_neurona.append(delta_peso)
                    deltas_pesos_capa.append(deltas_pesos_neurona)
                    deltas_sesgos_capa.append(delta_sesgo_neurona)
                    derivadas_pesos_neuronas.append(derivada_peso_neurona)
            _derivadas_pesos_neuronas = derivadas_pesos_neuronas
            _capa_neuronal = capa_neuronal
            deltas_pesos_red.insert(0,deltas_pesos_capa)
            deltas_sesgos_red.insert(0,deltas_sesgos_capa)
        return deltas_pesos_red,deltas_sesgos_red

    def aplicarDeltas(red_neuronal,deltas):

        for capa_neuronal in range(len(red_neuronal)):
            for neurona in range(len(red_neuronal[capa_neuronal])):
                for peso in range(len(red_neuronal[capa_neuronal][neurona].pesos)):
                    delta = deltas[0][capa_neuronal][neurona][peso]
                    red_neuronal[capa_neuronal][neurona].pesos[peso] -= red_neuronal.ALFA * delta
                delta = deltas[1][capa_neuronal][neurona]
                red_neuronal[capa_neuronal][neurona].sesgo -= red_neuronal.ALFA * delta

    def obtenerError(red_neuronal, set_entrenamiento):
        error_total = 0
        for _set in range(len(set_entrenamiento)):
            x, y = set_entrenamiento[_set]
            red_neuronal(x)
            for salida in range(len(y)):
                neurona = red_neuronal[-1][salida]
                error_total += neurona.funcionCoste(neurona.salida,y[salida])
        return error_total

    def entrenar(red_neuronal, set_entrenamiento,verbose=False):
        error = red_neuronal.obtenerError(set_entrenamiento)
        _iter = 0
        while error > red_neuronal.ERROR_MINIMO:
            x,y = random.choice(set_entrenamiento)
            deltas = red_neuronal.obtenerDeltas(x,y)
            red_neuronal.aplicarDeltas(deltas)
            error = red_neuronal.obtenerError(set_entrenamiento)
            _iter += 1
            if (_iter % 10000) == 0 and verbose:
                print(_iter,error)
        print(_iter,error)

red = RedNeuronal([
    CapaNeuronal([Neurona(),Neurona(),]),
    CapaNeuronal([Neurona(),]),
])

set_entrenamiento = [
    [[0,0],[0]],
    [[0,1],[1]],
    [[1,0],[1]],
    [[1,1],[0]],
]

red.entrenar(set_entrenamiento,True)

for _set in set_entrenamiento:
    print(red(_set[0]))
