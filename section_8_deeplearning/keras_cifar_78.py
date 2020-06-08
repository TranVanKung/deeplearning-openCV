from keras.datasets import cifar10
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from sklearn.metrics import classification_report

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape)
# plt.imshow(x_train[0])
# plt.show()

x_train = x_train/255
x_test = x_test/255

print(y_train)
y_cat_train = to_categorical(y_train, 10)
y_cat_test = to_categorical(y_test, 10)

# build model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(4, 4),
                 input_shape=(32, 32, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(filters=32, kernel_size=(4, 4),
                 input_shape=(32, 32, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten())
# 128, 256, 512
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop', metrics=['accuracy'])
model.summary()
model.fit(x_train, y_cat_train, verbose=1, epochs=1)

print(model.metrics_names)
print(model.evaluate(x_test, y_cat_test))
predictions = model.predict_classes(x_test)
print(classification_report(y_test, predictions))
