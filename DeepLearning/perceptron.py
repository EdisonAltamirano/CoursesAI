from random import random

class Perceptron():
    def __init__(self):
        self.weights = []
        self.dimension = 2
        self.bias = 0
        # self.train 
        self.test = []

    def initializeWeights(self):
        for d in range(self.dimension):
            self.weights.append(0)
    
    def createTestTrain(self, dataset):
        self.test = dataset[:10]
        self.train = dataset[10:]
    
    def algoritmoTrain(self, dataset):
        self.initializeWeights()
        self.createTestTrain(dataset)
        for it in range(len(self.test)):
            a = 0
            for d in range(self.dimension):
                a += self.weights[d]*self.test[it][d] + self.bias
            if dataset[it][2]*a <= 0:
                for d in range(self.dimension):
                    self.weights[d] = self.weights[d] + self.test[it][2]*self.test[it][d]
                    self.bias = self.bias + self.test[it][2]
        return self.weights, self.bias

    def algoritmoTest(self):
        a = 0
        for d in range(self.dimension):
            a += self.weights[d]

    def error(self):
        return 0


dataset20 = []
dataset60 = []
dataset110 = []

# Formula Ax1 + Bx2 + c (+ ruido)
def formula():
    a = 20
    b = 10
    c = 5
    x1 = random()
    x2 = random()
    ruido = random()
    dato = a*x1 + b*x2 + c + ruido
    return [x1,x2,dato]

def createDataSet(num, dataset):
    for i in range(num):
        data = formula()
        dataset.append(data)

# Genera un set de 20
createDataSet(20, dataset20)

# Genera un set de 60 
createDataSet(60, dataset60)

# Genera un set de 110
createDataSet(110, dataset110)

# Algoritmo de perceptrón
perceptron20 = Perceptron()
perceptron60 = Perceptron()
perceptron110 = Perceptron()
res20 = perceptron20.algoritmoTrain(dataset20)
res60 = perceptron60.algoritmoTrain(dataset60)
res110 = perceptron110.algoritmoTrain(dataset110)

print(res20)
#print(res60)
#print(res110)