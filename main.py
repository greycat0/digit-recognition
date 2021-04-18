from mnist import MNIST
import numpy as np

from layer import Layer
from helper import Helper
# Конфигурация
q = 0.097
Config = (784, 16, 16, 10)
np.random.seed(9)

# Создание нейронной сети по заданной конфигурации
neural_network = []
for i, amount in enumerate(Config):
    prev_layer = None
    if len(neural_network) > 0:
        prev_layer = neural_network[-1]
    neural_network.append(Layer(prev_layer))
    if prev_layer:
        neural_network[-1].W = np.random.sample((Config[i - 1], amount))
        neural_network[-1].B = np.random.sample(amount)

# Загрузка обучающих данных из файла
n = 300
mndata = MNIST('.')
images, labels = mndata.load_training()
X = np.array(images[:n]) / 255    # Матрица входных данных (n, 784)
answers = labels[:n]
R = np.zeros((n, 10))
for i, answer in enumerate(answers):
    R[i][answer] = 1                # Матрица правильных ответов (n, 10)

neural_network[0].Y = X

# Обучение
for _ in range(10000):
    neural_network[-1].activate()       # Активация сети, получение Y (n, 10)
    for layer in reversed(neural_network[1:]):
        if layer is neural_network[-1]:
            layer.E = 2 * (R - layer.Y)     # Матрица ошибок выходного слоя (n, 10)
            print(np.sum(abs(layer.E)))
        dB = layer.E * Helper.sigmoid_der(layer.Y)      # Матрица смещений (n, 10)
        dW = np.dot(layer.prev.Y.T, dB)     # Матрица изменений весов (число нейронов предыдущего слоя , число нейронов текущего слоя)
        layer.prev.E = np.dot(dB, layer.W.T) # Матрица ошибок предыдущего слоя (n, число нейронов предыдущего слоя)
        layer.W += q * dW   # Применение изменений весов
        #layer.B += np.sum(dB, axis=0)   # Применение изменений смещений

# Проверка
testImages, testLabels = mndata.load_testing()
testImages = np.array(testImages) / 255    # Матрица входных данных (n, 784)
for i in range(100):
    neural_network[0].Y = np.array(testImages[i])
    prediction = neural_network[-1].activate()
    result = np.argsort(prediction)[-1]
    print(f'Ответ сети: {result}, с вероятностью {prediction[result] * 100}%, правильный ответ: {testLabels[i]}')