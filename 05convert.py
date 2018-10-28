#!/usr/bin/python
#
# convert features by DNN
import numpy as np
import chainer
from chainer import Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList, cuda
import chainer.functions as F
import chainer.links as L
import pysptk as sptk
import pyworld as pw
from scipy.io import wavfile
import os
import sys
import time
import cupy as cp


gpu_device = 0
cuda.get_device(gpu_device).use()
xp = cuda.cupy

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
                    l2=myLSTM(in_size = n_units, out_size = n_units),
                    l3=myLSTM(in_size = 2*n_units, out_size = n_units),
                    l4=myLSTM(in_size = 2*n_units, out_size = n_units//2),
                    l5=L.Linear(n_units,dim))

        def __call__(self, x_data, y_data, dim=25):
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
    dim = 25
    n_units = 64

    model = VCDNN(dim,n_units)
    model.to_cpu()
    xp = np
    serializers.load_npz("model/mgc/vcmodel4.npz",model)
    model.to_gpu()
    xp = cuda.cupy

    # test data
    x = []
    datalist = []
    with open("conf_mgc/eval.list","r") as f:
        for line in f:
            line = line.rstrip()
            datalist.append(line)

    for d in datalist:
        with open("data/male/mgc/{}.mgc".format(d),"rb") as f:
            dat = np.fromfile(f,dtype="<f8",sep="")
            x.append(xp.array(dat.reshape(int(len(dat)/dim),dim)).astype(xp.float32))

    if not os.path.isdir("result"):
        os.mkdir("result")
    if not os.path.isdir("result/wav3layer"):
        os.mkdir("result/wav3layer")

    fs = 16000
    fftlen = 512
    alpha = 0.42
    for i in range(0,len(datalist)):
        outfile = "result/wav3layer/{}.wav".format(datalist[i])
        with open("data/male/f0/{}.f0".format(datalist[i]),"rb") as f:
            f0 = np.fromfile(f, dtype="<f8", sep="")
            f0 = 2.0 * f0
        with open("data/male/ap/{}.ap".format(datalist[i]),"rb") as f:
            ap = np.fromfile(f, dtype="<f8", sep="")
            ap = ap.reshape(int(len(ap)/(fftlen+1)),fftlen+1)
        model.to_cpu()
        xp = np
        y = model.get_predata(chainer.cuda.to_cpu(x[i]).astype(np.float32))
        y = np.array(y).astype(np.float64)
        sp = sptk.mc2sp(y, alpha, fftlen*2)

        owav = pw.synthesize(f0, sp, ap, fs)
        owav = np.clip(owav, -32768, 32767)
        wavfile.write(outfile, fs, owav.astype(np.int16))
