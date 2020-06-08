from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from sklearn.metrics import classification_report

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)
single_image = x_train[0]
# print(single_image)
# plt.imshow(single_image, cmap='gray_r')
# plt.show()

y_cat_test = to_categorical(y_test, 10)
y_cat_train = to_categorical(y_train, 10)
# print(y_cat_train)
# print(y_cat_train[0])

# print(single_image.max())
# print(single_image.min())
x_train = x_train/x_train.max()
x_test = x_test/x_test.max()
scaled_image = x_train[0]
# print(scaled_image.max())
x_train = x_train.reshape(60000, 28, 28, 1)
# print(x_train.shape)
x_test = x_test.reshape(10000, 28, 28, 1)

model = Sequential()
# convolution
model.add(Conv2D(filters=32, kernel_size=(4, 4),
                 input_shape=(28, 28, 1), activation='relu'))
# pooling
model.add(MaxPool2D(pool_size=(2, 2)))
# 2d -> 1d
model.add(Flatten())
# dense layers
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop', metrics=['accuracy'])
model.summary()

model.fit(x_train, y_cat_train, epochs=2)
print(model.evaluate(x_test, y_cat_test))

predictions = model.predict_classes(x_test)
print(classification_report(y_test, predictions))
