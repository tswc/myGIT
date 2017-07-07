import scipy.io as sio
import numpy as np
import TJ_pack as tj
import random as rd


np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop
from keras import regularizers


# filepath = r'C:\Users\TSWC\Google Drive\FA\Python_101\\'

data = np.load("FQ101_data.npy")
print data.shape

xIR, yIR = tj.data2x_y_logic(data, np.arange(0, 100.5, 0.5))
y_compre = tj.compresion(yIR, 4)
# np.set_printoptions(threshold=None)

print y_compre

print xIR.shape, yIR.shape
# print yIR
# rd.shuffle(xIR)
# rd.shuffle(yIR)

x_train, x_test = tj.Sep_TT(xIR, 0.2)
y_train, y_test = tj.Sep_TT(y_compre, 0.2)

print type(x_train[3000,0]), y_train.shape

# parameter for regularation
lamda = 0

model = Sequential([
    Dense(52, input_dim=202, kernel_constraint=regularizers.l2(lamda), kernel_initializer='random_uniform'),
    Activation('relu'),

    Dense(51, kernel_regularizer=regularizers.l2(lamda), kernel_initializer='random_uniform'),
    Activation('softmax')


])

model.compile(
     optimizer='Adam',
     loss='categorical_crossentropy',
     metrics=['accuracy']
)


print('Training ----------------------------------------------')

model.fit(xIR, yIR, epochs=10, batch_size=32)

print('\nTesting-----------------------------------------------')

loss, accuracy = model.evaluate(x_train,y_train)
print('Train loss:', loss, 'Train accuracy:', accuracy)

loss_t,accuracy_t = model.evaluate(x_test, y_test)
print('Test loss:',loss_t, 'Test accruracy:', accuracy_t)


