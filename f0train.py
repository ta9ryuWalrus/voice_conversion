#!/usr/bin/python
#
# train the model
import numpy as np
import os
import sys
import time
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle




def get_dataset(dim=1):
    x = []
    y = []
    datalist = []
    with open("conf_f0/train.list","r") as f:
        for line in f:
            line = line.rstrip()
            datalist.append(line)

    for d in datalist:
        print(d)
        with open("data/male/data/f0/{}.dat".format(d),"rb") as f:
            dat = np.fromfile(f,dtype="<f8",sep="")
            x.append(list(np.array(dat, dtype = np.float64)))
        with open("data/female/data/f0/{}.dat".format(d),"rb") as f:
            dat = np.fromfile(f,dtype="<f8",sep="")
            y.append(list(np.array(dat, dtype = np.float64)))
    return x, y


if __name__ == "__main__":
    x_data, y_data = get_dataset()
    #print('x_data:', x_data[0].shape)
    dim = 1
    N = len(x_data)
    x_data = np.array(x_data).reshape(N, -1)
    y_data = np.array(y_data).reshape(N, -1)
    x_train, x_test, y_train, y_test = train_test_split(x_data[0], y_data[0], test_size = 0.1)


    x_train = np.array(x_train).reshape(-1, 1)
    y_train = np.array(y_train).reshape(-1, 1)
    x_test = np.array(x_test).reshape(-1, 1)
    y_test = np.array(y_test).reshape(-1, 1)

    """
    print('x_train:', x_train.shape)
    print('y_train:', y_train.shape)
    print('x_test:', x_test.shape)
    print('y_test:', y_test.shape)
    """
    log_x_train = np.log(np.clip(x_train, 1e-8, x_train))
    log_y_train = np.log(np.clip(y_train, 1e-8, y_train))
    log_x_test = np.log(np.clip(x_test, 1e-8, x_test))
    log_y_test = np.log(np.clip(y_test, 1e-8, y_test))

    """x_trainとy_trainは結合してX?"""
    model = LinearRegression(n_jobs = -1)
    model.fit(log_x_train, log_y_train)
    print('score:', model.score(log_x_test, log_y_test))
    print('coef:', model.coef_)
    print('intercept', model.intercept_)

    if not os.path.isdir("model_f0"):
        os.mkdir("model_f0")
    filename = 'model_f0/f0model.sav'
    pickle.dump(model, open(filename, 'wb'))
