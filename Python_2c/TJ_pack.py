import numpy as np
import scipy.io as sio
import math

def Comp2RI(x):
    Real = np.real(x)
    Imag = np.imag(x)
    IR_mat = np.column_stack((Real,Imag))
    return IR_mat

def to_logic(x, num):
    y = np.zeros((x.shape[0], len(num)))
    for i in range(x.shape[0]):
        for j in range(len(num)):
            if x[i] == num[j]:
                y[i, j] = 1.0
            else:
                y[i, j] = 0.0
    return y


# def mat2py(filepath, filename):
#     data_IR = None
#     for i in range(len(filename)):
#         file_ = filepath + filename[i]
#
#         data = sio.loadmat(file_)
#
#         b = data['Out_Matrix']
#         yAbstand = b[:(b.shape[0] - 1), b.shape[1] - 1]
#         xAbstand = b[:(b.shape[0] - 1), :b.shape[1] - 1]
#         xIR = Comp2RI(xAbstand)
#         yIR = Comp2RI(yAbstand)
#         xyIR = np.column_stack((xIR, yIR))
#         if data_IR == None:
#             data_IR = xyIR
#         else:
#             data_IR = np.row_stack((data_IR, xyIR))
#
#
#     return data_IR

# get all the date from mat file into matrix, and combine them in one

def mat2py(filepath, filename):
    data_IR = None
    for i in range(len(filename)):
        file_ = filepath + filename[i]

        data = sio.loadmat(file_)


        b = data['Out_Matrix']
        yAbstand = b[:(b.shape[0] - 1), b.shape[1] - 1]
        xAbstand = b[:(b.shape[0] - 1), :b.shape[1] - 1]
        xIR = Comp2RI(xAbstand)
        yIR = Comp2RI(yAbstand)
        xyIR = np.column_stack((xIR, yIR))
        if data_IR == None:
            data_IR = xyIR
        else:
            data_IR = np.row_stack((data_IR, xyIR))


    return data_IR

def data2x_y_logic(m, n_range=None, Dmedian=None):
    xIR = m[:, :m.shape[1] - 2]
    yIR = m[:, (m.shape[1] - 2):(m.shape[1] - 1)]
    if n_range != None:
        y_logic = to_logic(yIR, n_range)
    elif Dmedian != None:
        y_logic = class2sep(yIR, Dmedian)
    else:
        y_logic = None
    return xIR, y_logic, yIR


def data2x_y_logic2(m, n_range=None, Dmedian=None):
    xIR = m[:, :m.shape[1] - 2]
    yIR = m[:, (m.shape[1] - 2):(m.shape[1] - 1)]
    if n_range != None:
        y_logic = to_logic(yIR, n_range)
    elif Dmedian != None:
        y_logic = class2sep(yIR, Dmedian)
    else:
        y_logic = None
    return xIR, y_logic, yIR


def Activation(x):

    sigmpod = 1./(1 + np.exp(-x))
    return sigmpod

def add_bias(X):
    return np.column_stack((np.ones(X.shape[0]), X))

def theta_init(m, n):
    # type: (object, object) -> object
    epsilon = 0.12
    W = np.random.random(size= (m, n))*2*epsilon - epsilon
    return W

def Sep_TT(data, test_split):
    num = data.shape[0]
    if not 0 <= test_split < 1:
        raise ValueError("sample larger than population")
    else:
        index_split = int((num+1)*(1 - test_split))
        return data[0:index_split, :], data[index_split:data.shape[0], :]


def compresion(data, rate):
    range1 = data.shape[1]
    num = data.shape[0]
    range_Comp = math.ceil(float(range1)/rate)
    print range_Comp

    data_comp = np.zeros((num, int(range_Comp)))
    for i in range(num):
        for j in range(range1):
            if data[i,j] == 1:
                data_comp[i, int(j/rate)] = 1.0

    return data_comp


def class2sep(data, judge):
    arr = np.where(data>judge, 1, 0)
    return arr

# def cost_F(x, y, theta, lamda):
#
#     xIR_bias = add_bias(x)
#
#     for
#
#     J = Activation(xIR_bias*theta.T)
