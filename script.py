import pandas as pd
from keras.utils import to_categorical
import numpy as np

#carregando dados de treino e teste
x_train = pd.read_csv('train.csv')
x_test = pd.read_csv('test.csv')
y_train = x_train[['label']]
x_train = x_train.drop(['label'], axis=1)

#classificação de cada dígito 42000 imagens 10 possiveis classes
y_train = to_categorical(y_train)
y_train.shape

#fazendo o reshape para 28x28 pixels
x_train = np.array(x_train).reshape(42000, 28, 28, 1)
x_test = np.array(x_test).reshape(28000, 28, 28, 1)


from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

model = Sequential()

#adicionando as camadas na rede
model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D())
model.add(Conv2D(64, kernel_size=3, activation='relu',padding='same'))
model.add(MaxPooling2D())
model.add(Conv2D(128, kernel_size=3, activation='relu',padding='same'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#treinando o modelo
model.fit(x_train, y_train, epochs=10)

prediction = model.predict(x_test)

results = np.argmax(prediction, axis=1)
results = pd.Series(results,name="Label")
results


