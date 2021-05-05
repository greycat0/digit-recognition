import time
from mnist import MNIST
import numpy as np

from layer import Layer
from helper import Helper
from paint import Paint

try:
    file = open('model', 'rb')
    neural_network = np.load(file, allow_pickle=True)
    file.close()
except FileNotFoundError:
    # Конфигурация
    q = 0.097
    Config = (784, 16, 16, 10)
    np.random.seed(16)

    # Создание нейронной сети по заданной конфигурации
    neural_network = []
    for i, amount in enumerate(Config):
        prev_layer = None
        if len(neural_network) > 0:
            prev_layer = neural_network[-1]
        neural_network.append(Layer(prev_layer))
        if prev_layer:
            neural_network[-1].W = np.random.sample((Config[i - 1], amount))

    # Загрузка обучающих данных из файла
    n = 5000
    b = 500 
    mndata = MNIST('.')
    images, labels = mndata.load_training()
    images = np.array(images[:n]) / 255  # Матрица входных данных (n, 784)
    answers = np.zeros((n, 10))
    for i, label in enumerate(labels[:n]):
        answers[i][label] = 1  # Матрица правильных ответов (n, 10)

    # Обучение
    for X, R, i in [(images[i:i + b], answers[i:i + b], i) for i in range(0, len(images), b)]:
        neural_network[0].Y = X
        print(f'batch {i}')
        time.sleep(1)
        for _ in range(100000):
            neural_network[-1].activate()  # Активация сети, получение Y (n, 10)
            for layer in reversed(neural_network[1:]):
                if layer is neural_network[-1]:
                    layer.E = 2 * (R - layer.Y)  # Матрица ошибок выходного слоя (n, 10)
                    error = np.sum(abs(layer.E))
                    print(error)
                dB = layer.E * Helper.sigmoid_der(layer.Y)  # Матрица смещений (n, 10)
                dW = np.dot(layer.prev.Y.T, dB)  # Матрица изменений весов (число нейронов предыдущего слоя , число нейронов текущего слоя)
                layer.prev.E = np.dot(dB, layer.W.T)  # Матрица ошибок предыдущего слоя (n, число нейронов предыдущего слоя)
                layer.W += q * dW  # Применение изменений весов
            if error < 100:
                break
    # Сохранение модели в файл
    neural_network[0].Y = None
    file = open('model', 'wb')
    np.save(file, neural_network)
    file.close()


# Проверка
def predict(event):
    image = [0] * 784
    canvas = event.widget
    for item in canvas.find_all():
        _, _, x, y = canvas.coords(item)
        i = int(y // 10 * 28 + x // 10)
        image[i] = 1
        if i > 0:
            image[i-1] = 1
        if i < 784:
            image[i+1] = 1
    print(MNIST.display([x * 255 for x in image]))
    neural_network[0].Y = np.array(image)
    prediction = neural_network[-1].activate()
    result = np.argsort(prediction)[-1]
    print(f'Ответ сети: {result}, с вероятностью {prediction[result] * 100}%')


Paint(predict)
