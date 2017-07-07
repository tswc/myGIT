

import numpy as np
import TJ_pack as tj
import random as rd
import matplotlib.pyplot as plt

np.random.seed(1337)  # for reproducibility
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop
from keras import regularizers

filepath = r'C:\Users\TSWC\Google Drive\FA\data_for_python\\'

test = np.load("FQ101_data.npy")

rd.shuffle(test)

x_t, y_t, yIR = tj.data2x_y_logic(test, Dmedian=50)
# print x_t.shape,y_t.shape
# mask = (yIR > 48.5)&(yIR < 51.5)
# index_Del = np.where(mask)
# x_t = np.delete(x_t,np.s_[index_Del[0]],axis=0)
# y_t = np.delete(y_t,np.s_[index_Del[0]],axis=0)
# print x_t.shape,y_t.shape
# x_train, x_test = tj.Sep_TT(x_t, 0.2)
# y_train, y_test = tj.Sep_TT(y_t, 0.2)
# y_t = tj.compresion(y_t, 4)
# y_test = tj.compresion(y_test, 4)

lamda1 = 0
lamda2 = 0
lamda3 = 0
aa= 0.01
num_layers = range(1, 6)

for l1 in range(1,5):
    for l2 in range(1,5):
        for l3 in range(1,5):
            Hlayer1 = 128*l1
            Hlayer2 = 128*l2
            Hlayer3 = 128*l3

            model = Sequential([
                Dense(256, input_dim=202, kernel_regularizer=regularizers.l1(0), kernel_initializer='random_uniform'),

                Activation('relu'),

                Dense(Hlayer1, kernel_regularizer=regularizers.l1_l2(0.001), kernel_initializer='random_uniform'),

                Activation('relu'),

                Dense(Hlayer2, kernel_regularizer=regularizers.l1_l2(0), kernel_initializer='random_uniform'),

                Activation('relu'),

                Dense(Hlayer3, kernel_regularizer=regularizers.l1_l2(0), kernel_initializer='random_uniform'),
                Activation('relu'),


                Dense(1, kernel_regularizer=regularizers.l1(0), kernel_initializer='random_uniform'),
                Activation('sigmoid')
                ])
                # kernel_initializer = 'random_uniform',

            # Another way to define your optimizer
            rmsprop  = RMSprop(lr=0.01, rho=0.9, epsilon=1e-08, decay=0.0)

            # We add metrics to get more results you want to use
            model.compile(
                # optimizer=rmsprop,
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'],
            )



            print('Training -----------------------')
            # Another way to train the model
            # model.fit(X_train, y_train, epochs=100, batch_size=64)
            Epoch = 150
            history = model.fit(x_t, y_t, epochs=Epoch, batch_size=64, validation_split=0.2)

            # print('\nTesting-----------------------')
            # #Evaluate the model with the matrics we defined earliea
            # loss, accuracy = model.evaluate(x_test, y_test)
            # print('test loss', loss)
            # print('test accuracy', accuracy)




            plt.figure(figsize=[10, 10])

            p_acc = plt.subplot(2, 3, (1, 2))
            plt.plot(history.history['acc'])
            plt.plot(history.history['val_acc'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            text = str(3) + ' Hiddenlayers\n' \
                            'layer1----------'+ str(Hlayer1) + ' neurals\n' \
                            'layer2----------'+ str(Hlayer2) + ' neurals\n' \
                            'layer3----------'+ str(Hlayer3) + ' neurals'
            # plt.figtext(0.65, 0.74, text)



            xy_position = (Epoch-1, history.history['acc'][-1])

            xyt_position = (Epoch-1, history.history['acc'][-1]+0.08)

            print xy_position, xyt_position
            p_acc.annotate(str(round(history.history['acc'][-1], 3)), xy=xy_position, xytext=xyt_position, arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))


            xy_position = (Epoch-1, history.history['val_acc'][-1])

            xyt_position = (Epoch-1, history.history['val_acc'][-1]-0.08)

            print xy_position, xyt_position
            p_acc.annotate(str(round(history.history['val_acc'][-1], 3)), xy=xy_position, xytext=xyt_position, arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

            # plt.savefig('figure1.png')
            # plt.show()

            p_loss = plt.subplot(2, 3, (4, 5))
            plt.plot(history.history['loss'])
            print type(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.figtext(0.65, 0.51, text)


            xy_position = (Epoch-1, history.history['loss'][-1])

            xyt_position = (Epoch-1, history.history['loss'][-1]-0.4)

            print xy_position, xyt_position
            p_loss.annotate(str(round(history.history['loss'][-1], 3)), xy=xy_position, xytext=xyt_position, arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))




            xy_position = (Epoch-1, history.history['val_loss'][-1])

            xyt_position = (Epoch-1, history.history['val_loss'][-1]+0.4)

            print xy_position, xyt_position
            p_loss.annotate(str(round(history.history['val_loss'][-1], 3)), xy=xy_position, xytext=xyt_position, arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
            plt.savefig('figure' + str(l1) + str(l2) + str(l3)+'.png')


# plt.show()