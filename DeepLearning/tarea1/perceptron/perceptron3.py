import numpy as np
import random

class Dataset():
    def __init__(self, num = 20):
        self.num = num
        self.dataset = []
        self.features = []
        self.labels = []
        self.testFeatures = []
        self.testLabels = []
        self.trainFeatures = []
        self.trainLabels = []

    # Formula Ax1 + Bx2 + b (+ ruido)
    def formula(self):
        a = 4
        b = -2.5
        c = 1.8
        x1 = random.random()
        x2 = random.random()
        ruido = random.random()
        dato = a*x1 + b*x2 + c + ruido
        return [x1,x2], dato

    def createDataSet(self):
        for i in range(self.num):
            features, label = self.formula()
            self.features.append(features)
            self.labels.append(label)

    def createTestTrain(self):
        self.testFeatures = self.features[:10]
        self.testLabels = self.labels[:10]
        self.trainFeatures = self.features[10:]
        self.trainLabels = self.labels[10:]

    def start(self):
        self.createDataSet()
        self.createTestTrain()

# CODIGO DE PERCEPTRON
def score(weights, bias, features):
    return np.dot(features, weights) + bias

def step(x):
    return x

# No tenemos predicción por ser el caso de una regresión donde tenemos un valor continuo como resultado
def prediction(weights, bias, features):
    return step(score(weights, bias, features))

def error(weights, bias, features, label):
    pred = prediction(weights, bias, features)
    if pred == label:
        return 0
    else:
        return np.abs(label - score(weights, bias, features))

def mean_perceptron_error(weights, bias, features, labels):
    total_error = 0
    for i in range(len(features)):
        total_error += error(weights, bias, features[i], labels[i])
    return total_error/len(features)

def perceptron_trick(weights, bias, features, label, learning_rate = 0.01):
    pred = prediction(weights, bias, features)
    for i in range(len(weights)):
        weights[i] += (label-pred)*features[i]*learning_rate
    bias += (label-pred)*learning_rate
    return weights, bias

def perceptron_algorithm(features, labels, learning_rate = 0.01, epochs = 10000):
    weights = [0.0 for i in range(len(features[0]))]
    bias = 0.0
    errors = []
    for epoch in range(epochs):
        error = mean_perceptron_error(weights, bias, features, labels)
        errors.append(error)
        i = random.randint(0, len(features) - 1)
        weights, bias = perceptron_trick(weights, bias, features[i], labels[i])
    return weights, bias, errors

def test(weights, bias, features, labels):
    predictions = prediction(weights, bias, features)
    error = 0
    for i in np.abs(labels - predictions):
        error += i
    return error/len(predictions)
    

def setup(num):
    dataset = Dataset(num)
    dataset.start()
    features = np.array(dataset.trainFeatures)
    labels = np.array(dataset.trainLabels)
    weights, bias, error = perceptron_algorithm(features, labels)
    featuresTest = np.array(dataset.testFeatures)
    labelsTest = np.array(dataset.testLabels)
    errorTest = test(weights, bias, featuresTest, labelsTest)
    return weights, bias, errorTest




vals = [20, 60, 110]
for val in vals:
    print("Corriendo con ",val," datos")
    print("TRAIN --------------------------------------------------------")
    print("Valores esperados: [4, -2.5] 1.8")
    w, b, eT = setup(val)
    print("Valores obtenidos: ", w, b)
    print("Diferencias:", 4 - w[0], -2.5-w[1], 1.8-b)
    print("%: ", ((4 - w[0])/4)*100,"%", ((-2.5-w[1])/(-2.5))*100,"%", ((1.8-b)/1.8)*100,"%")
    print("TEST ---------------------------------------------------------")
    print("Error de prueba:", eT)
    print("\n")



