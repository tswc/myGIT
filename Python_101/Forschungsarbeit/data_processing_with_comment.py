
# Here is the place to import all the package I need

import numpy as np
import TJ_pack as tj               # this my own package to realize some functions which I always use
import random as rd
import matplotlib.pyplot as plt

np.random.seed(1337)  # for reproducibility
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import RMSprop
from keras import regularizers




# import the data
filepath = r'C:\Users\TSWC\Google Drive\FA\data_for_python\\'

test = np.load("FQ101_data.npy")

# pre-processing of the data
# divide the data into X and Y, X is the input, Y is the output also the label. Y was also changed into vector.
x_t, y_t, yIR = tj.data2x_y_logic(test, np.arange(0, 100.5, 0.5))

# divide the data into 2 sets. One is for training , the other is for testing
x_train, x_test = tj.Sep_TT(x_t, 0.2)
y_train, y_test = tj.Sep_TT(y_t, 0.2)

# Change the output from 201 dimensions into 51
y_train = tj.compresion(y_train, 4)
y_test = tj.compresion(y_test, 4)

# Use a variable to save the Classes Number of label.
y_t_class = np.where(y_test==1)




# Basic setting for number of neurons in each layer
lamda1 = 0
lamda2 = 0
lamda3 = 0



# Here is how I build the model.
# I use three loops to see how will the number of neurons will influense the performence of the model

for l1 in range(1,5):
    for l2 in range(1,5):
        for l3 in range(1,5):
            Hlayer1 = 128*l1
            Hlayer2 = 128*l2
            Hlayer3 = 128*l3

			
			# 5 layers in the model, included 1 input layer, 3 hidden layer, 1 output layer

            model = Sequential([
                Dense(256, input_dim=202,kernel_initializer='random_uniform'),   # 202 input, 256 output, In this layer the input and output are fixed. The initialization of this layer is using "random uniform"

                Activation('relu'),  # every layer need a activation function to add some nolinear part in the net

                Dense(Hlayer1 , kernel_regularizer=regularizers.l2(0.001), kernel_initializer='random_uniform'), # use regularation with l2 method

                Activation('relu'),
                # Dropout(0.15),
                Dense(Hlayer2,  kernel_regularizer=regularizers.l2(0.001), kernel_initializer='random_uniform'),

                Activation('relu'),
                # Dropout(0.15),

                Dense(Hlayer3,kernel_regularizer=regularizers.l2(0.001),  kernel_initializer='random_uniform'),
                Activation('relu'),
                # Dropout(0.15),

                Dense(51, kernel_initializer='random_uniform'),  # output layer , 51 Output.
                Activation('softmax')
                ])
                # kernel_initializer = 'random_uniform',


            # this tell the program how to train our net.
            model.compile(
                # optimizer=rmsprop,
                optimizer='adam',    # use this algorithm to find the minimum
                loss='categorical_crossentropy',  # use this algorithm to define the loss function
                metrics=['accuracy'],	# what is showing in the command window
            )


			# the training part
			# we will train our net for 200 times
			# the batch size will be 64. That means we will treat every 64 example as a entirety to calculate the Updating Information.
			# validation_split means how many examples we will use  as the cross validation set
            print('Training -----------------------')
            Epoch = 200
            history = model.fit(x_train, y_train, epochs=Epoch, batch_size=64, validation_split=0.2)

            print('\nTesting-----------------------')
			# evaluste the model with the test set
            loss, accuracy = model.evaluate(x_test, y_test)
            print('test loss', loss)
            print('test accuracy', accuracy)

			
			
			
			# use predict to find how this model thw test set predict
            y = model.predict_classes(x_test)
			# the difference between the the label and our prediction
            diff = abs(y_t_class[1] - y)
			# calculate the variance
            meanerror = np.sum(diff) / float(y.shape[0])*2

			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			# The part below is to plot some picture to give us a visuell feeling about the performance

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
                            'layer3----------'+ str(Hlayer3) + ' neurals\n' \
                            'test accuracy:'+ str(accuracy)+'\n   testloss:'+ str(loss)+ \
                            '\n error of mm:' + str(meanerror) + \
                            '\n maximum error:' + str(np.max(diff)*2)
            # plt.figtext(0.65, 0.74, text)



            xy_position = (Epoch-1, history.history['acc'][-1])

            xyt_position = (Epoch-1, history.history['acc'][-1]+0.2)

            print xy_position, xyt_position
            p_acc.annotate(str(round(history.history['acc'][-1], 3)), xy=xy_position, xytext=xyt_position, arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))


            xy_position = (Epoch-1, history.history['val_acc'][-1])

            xyt_position = (Epoch-1, history.history['val_acc'][-1]-0.2)

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

            xyt_position = (Epoch-1, history.history['loss'][-1]-1)

            print xy_position, xyt_position
            p_loss.annotate(str(round(history.history['loss'][-1], 3)), xy=xy_position, xytext=xyt_position, arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))




            xy_position = (Epoch-1, history.history['val_loss'][-1])

            xyt_position = (Epoch-1, history.history['val_loss'][-1]+1)

            print xy_position, xyt_position
            p_loss.annotate(str(round(history.history['val_loss'][-1], 3)), xy=xy_position, xytext=xyt_position, arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
            plt.savefig('figure' + str(l1) + str(l2) + str(l3)+'.png')


# plt.show()