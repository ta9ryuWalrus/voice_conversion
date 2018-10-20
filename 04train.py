#!/usr/bin/python
#
# train the model
import numpy as np
import chainer
from chainer import Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList, cuda
from chainer.training import extensions
from chainer.datasets import split_dataset_random
import chainer.functions as F
import chainer.links as L
import os
import sys
import time
import cupy as cp


gpu_device = 0
cuda.get_device(gpu_device).use()
xp = cuda.cupy

def get_dataset(dim=25):
    x = []
    y = []
    datalist = []
    with open("conf_mgc/train.list","r") as f:
        for line in f:
            line = line.rstrip()
            datalist.append(line)

    for d in datalist:
        print(d)
        with open("data/male/data/mgc/{}.dat".format(d),"rb") as f:
            dat = np.fromfile(f,dtype="<f8",sep="")
            x.append(xp.array(dat.reshape(int(len(dat)/dim),dim)).astype(xp.float32))
        with open("data/female/data/mgc/{}.dat".format(d),"rb") as f:
            dat = np.fromfile(f,dtype="<f8",sep="")
            y.append(xp.array(dat.reshape(int(len(dat)/dim),dim)).astype(xp.float32))
    return x, y

class myLSTM(L.NStepBiLSTM):
    def __init__(self, in_size, out_size, dropout = 0.5):
        n_layers = 2
        super(myLSTM, self).__init__(n_layers, in_size, out_size, dropout)
        self.state_size = out_size
        self.reset_state()

    def to_cpu(self):
        super(myLSTM, self).to_cpu()
        if self.cx is not None:
            self.cx.to_cpu()
        if self.hx is not None:
            self.hx.to_cpu()

    def to_gpu(self, device = 0):
        super(myLSTM, self).to_gpu(device)
        if self.cx is not None:
            self.cx.to_gpu(device)
        if self.hx is not None:
            self.hx.to_gpu(device)

    def set_state(self, cx, hx):
        assert isinstance(cx, Variable)
        assert isinstance(hx, Variable)
        cx_ = cx
        hx_ = hx

        if self.xp == np:
            cx_.to_cpu()
            hx_.to_cpu()
        else:
            cx_.to_gpu()
            hx_.to_gpu()
        self.cx = cx_
        self.hx = hx_

    def reset_state(self):
        self.cx = self.hx = None

    def __call__(self, xs):
        batch = 1
        if self.hx is None:
            self.hx = Variable(xp.zeros((self.n_layers * 2, batch, self.state_size), dtype = xp.float32))
        if self.cx is None:
            self.cx = Variable(xp.zeros((self.n_layers * 2, batch, self.state_size), dtype = xp.float32))
        x = []
        x.append(xs)
        hy, cy, ys = super(myLSTM, self).__call__(self.hx, self.cx, x)
        self.hx, self.cx = hy, cy
        return ys[0]


class VCDNN(Chain):
        def __init__(self, dim=25, n_units=64):
            super(VCDNN, self).__init__(
                    l1=L.Linear(dim, n_units),
                    l2=myLSTM(in_size = n_units, out_size = n_units, dropout = 0.5),
                    l3=myLSTM(in_size = 2*n_units, out_size = n_units, dropout = 0.5),
                    l4=myLSTM(in_size = 2*n_units, out_size = n_units//2, dropout = 0.5),
                    l5=L.Linear(n_units,dim)
                    )

        def __call__(self, x_data, y_data,dim=25):
                x = Variable(x_data.astype(xp.float32).reshape(len(x_data),dim))
                y = Variable(y_data.astype(xp.float32).reshape(len(y_data),dim))
                h3 = self.predict(x)
                return F.mean_squared_error(h3, y)

        def predict(self, x):
            h1 = F.relu(self.l1(x))
            h2 = self.l2(h1)
            h3 = self.l3(h2)
            h4 = self.l4(h3)
            h5 = self.l5(h4)
            return h5

        def get_predata(self, x):
            return self.predict(Variable(x.astype(xp.float32))).data


if __name__ == "__main__":
    x_train, y_train = get_dataset()

    batch_size = 1
    n_epoch = 50
    dim = 25
    #n_units = 128
    n_units = 64
    N = len(x_train)

    model = VCDNN(dim,n_units)
    #serializers.load_npz("model/mgc/vcmodel2.npz", model)
    model.l2.reset_state()
    model.l3.reset_state()
    model.l4.reset_state()
    model.to_gpu()

    optimizer = optimizers.Adam()
    optimizer.setup(model)


    # loop
    losses = []
    sum_loss = 0
    for epoch in range(1, n_epoch + 1):
        sum_loss = 0

        indices = np.random.permutation(N)
        for i in range(0, N):
            #x_batch = x_train[i*batch_size : (i+1)*batch_size]
            #y_batch = y_train[i*batch_size : (i+1)*batch_size]
            x_batch = x_train[indices[i]]
            y_batch = y_train[indices[i]]
            model.zerograds()
            model.l2.reset_state()
            model.l3.reset_state()
            model.l4.reset_state()
            loss = model(x_batch, y_batch, dim)
            sum_loss += loss.data
            loss.backward()
            optimizer.update()
            average_loss = sum_loss / N
            losses.append(average_loss)
            if (i // 50 == 0) and (i != 0):
                model.to_cpu()
                if not os.path.isdir("model/mgc"):
                    os.mkdir("model/mgc")
                serializers.save_npz("model/mgc/vcmodel2.npz", model)
                model.to_gpu()

            print("epoch: {}/{}  data: {}/{}  sum_loss: {}".format(epoch, n_epoch, i, N, sum_loss))


    model.to_cpu()
    if not os.path.isdir("model/mgc"):
        os.mkdir("model/mgc")
    serializers.save_npz("model/mgc/vcmodel2.npz",model)
