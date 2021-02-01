from keras.models import Sequential
from keras.layers.core import Dense
import numpy as np
import time

def create_mlp():

    model = Sequential()
    model.add(Dense(16, input_dim=6, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(8, activation="relu"))
    model.add(Dense(1, activation="linear"))

    return model


trainX = np.load('trainX.npy')
trainY = np.load('trainY.npy')
testX = np.load('testX.npy')
testY = np.load('testY.npy')

model = create_mlp()
#model.compile(loss="mean_absolute_percentage_error")
model.compile(loss="mean_squared_error")

# train the model
model.fit(trainX, trainY, validation_data=(testX, testY), epochs=500)

# make predictions on the testing data
preds = model.predict(testX)

preds2 = preds[:,0]
diff = preds2 - testY
perc = 100 - (preds2/testY)*100

z1 = np.random.rand(3000000, 6)

start = time.time()

z = model.predict(z1)

end = time.time()

print(end - start)