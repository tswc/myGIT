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

test = test[0:14000,:]
# rd.shuffle(test)

x_t, y_t, y_num = tj.data2x_y_logic(test, n_range=np.arange(0, 100.5, 0.5))
print np.max(np.max(x_t,axis=0))