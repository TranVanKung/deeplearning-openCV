import matplotlib.pyplot as plt
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Conv2D, MaxPooling2D, Dense
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

cat4 = cv2.imread('../DATA/CATS_DOGS/train/CAT/4.jpg')
cat4 = cv2.cvtColor(cat4, cv2.COLOR_BGR2RGB)
print(cat4.shape)
# plt.imshow(cat4)
# plt.show()

dog = cv2.imread('../DATA/CATS_DOGS/train/DOG/2.jpg')
dog = cv2.cvtColor(dog, cv2.COLOR_BGR2RGB)
print(dog.shape)
# plt.imshow(dog)
# plt.show()

image_gen = ImageDataGenerator(
    rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, rescale=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
# plt.imshow(image_gen.random_transform(dog))
# plt.show()

# print(image_gen.flow_from_directory('../DATA/CATS_DOGS/train'))

input_shape = (150, 150, 3)
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3),
                 input_shape=input_shape, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3, 3),
                 input_shape=(150, 150, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3, 3),
                 input_shape=(150, 150, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])
model.summary()

batch_size = 16
train_image_gen = image_gen.flow_from_directory(
    '../DATA/CATS_DOGS/train', target_size=input_shape[:2], batch_size=batch_size, class_mode='binary')
test_image_gen = image_gen.flow_from_directory(
    '../DATA/CATS_DOGS/test', target_size=input_shape[:2], batch_size=batch_size, class_mode='binary')
print(train_image_gen.class_indices)
# results = model.fit_generator(
#     train_image_gen, epochs=1, steps_per_epoch=150, validation_data=test_image_gen, validation_steps=12)
# results.history['acc']

# new_model = load_model('../DATA/cat_dog_100epochs.h5')
dog_file = '../DATA/CATS_DOGS/test/DOG/10005.jpg'
dog_img = image.load_img(dog_file, target_size=(150, 150))
dog_img = image.img_to_array(dog_img)
print(dog_img.shape)
dog_img = np.expand_dims(dog_img, axis=0)
print(dog_img.shape)
dog_img = dog_img / 255
print(model.predict_classes(dog_img))
print(model.predict(dog_img))
