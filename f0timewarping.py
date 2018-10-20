#!/usr/bin/env python

import os
import sys
import array

from dtw import dtw
import numpy as np
import pysptk as sptk

def distfunc(x,y):
    # Euclid distance except first dim
    return np.linalg.norm(x[1:]-y[1:])

srcspk = "male"
tgtspk = "female"

f0list = os.listdir("data/{}/f0".format(srcspk))

if not os.path.isdir("data/{}/data/f0".format(srcspk)):
    os.mkdir("data/{}/data/f0".format(srcspk))
if not os.path.isdir("data/{}/data/f0".format(tgtspk)):
    os.mkdir("data/{}/data/f0".format(tgtspk))

dim = 1 # mgc dim + 1
for mf in f0list:
    print(mf)
    bn, _ = os.path.splitext(mf)
    srcfile = "data/{}/f0/{}".format(srcspk,mf)
    tgtfile = "data/{}/f0/{}".format(tgtspk,mf)

    with open(srcfile,"rb") as f:
        x = np.fromfile(f, dtype="<f8", sep="")
        x = x.reshape(int(len(x)/dim),dim)
    with open(tgtfile,"rb") as f:
        y = np.fromfile(f, dtype="<f8", sep="")
        y = y.reshape(int(len(y)/dim),dim)
    print("framelen: (x,y) = {} {}".format(len(x),len(y)))
    _,_,_, twf = dtw(x,y,distfunc)
    srcout = "data/{}/data/f0/{}.dat".format(srcspk,bn)
    tgtout = "data/{}/data/f0/{}.dat".format(tgtspk,bn)

    with open(srcout,"wb") as f:
        x[twf[0]].tofile(f)
    with open(tgtout,"wb") as f:
        y[twf[1]].tofile(f)
