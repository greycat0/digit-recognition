import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from mnist import MNIST
from paint import Paint

input_shape = (28, 28, 1)

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ]
)

model.summary()

batch_size = 128
epochs = 1

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

def predict(event):
    image = [[[0] for i in range(28)] for j in range(28)]
    canvas = event.widget
    for item in canvas.find_all():
        _, _, x, y = canvas.coords(item)
        x, y = int(x // 10), int(y // 10)
        image[y][x][0] = 1
        # if image[y+1][x][0] == 0:
        #     image[y+1][x][0] = 0.6
        # if image[y-1][x][0] == 0:
        #     image[y-1][x][0] = 0.6
        # if image[y][x+1][0] == 0:
        #     image[y][x+1][0] = 0.6
        # if image[y][x+1][0] == 0:
        #     image[y][x+1][0] = 0.6
    print(MNIST.display([j[0] * 255 for i in image for j in i]))
    # plt.imshow(image, cmap=plt.cm.binary)
    # plt.show()
    prediction = model.predict(np.expand_dims(image, axis=0))
    result = np.argmax(prediction)
    print(f'Ответ сети: {result} c вероятностью {prediction[0][result] * 100}%')


Paint(predict)
