import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import load_model

data = genfromtxt('../DATA/bank_note_data.txt', delimiter=',')
# print(data)
labels = data[:, 4]
# print(labels)
features = data[:, 0:4]
# print(features)

X = features
y = labels
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)
print(len(X_train))
print(len(X_test))
# print(y_train)

scaler_object = MinMaxScaler()
scaler_object.fit(X_train)
scaled_X_train = scaler_object.transform(X_train)
scaled_X_test = scaler_object.transform(X_test)
# print(scaled_X_train.max())
# print(scaled_X_train.min())
# print(scaled_X_train)
# print(X_train)


model = Sequential()
model.add(Dense(4, input_dim=4, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])
model.fit(scaled_X_train, y_train, epochs=50, verbose=2)

# print(model.predict_classes(scaled_X_test))
print(model.metrics_names)
predictions = model.predict_classes(scaled_X_test)
confusion_matrix(y_test, predictions)
print(classification_report(y_test, predictions))

model.save('my_model.h5')
new_model = load_model('my_model.h5')
new_model.predict_classes(scaled_X_test)
