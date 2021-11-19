#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 09:12:52 2021

@author: shanjiang
"""

# https://www.youtube.com/watch?v=QUT1VHiLmmI

import numpy as np


if __name__ == '__main__':
    # basic
    # never use a = b, but use a = b.copy()
    a = np.ones((5, 5))
    a10 = np.ones(10)
    a1 = a.copy()
    a1[0] = 100
    # randint for integers, rand for float 0~1
    b = np.zeros((5, 5))
    c = np.random.randint(7, size=(1,3))
    c1 = np.repeat(c, 2, axis=0)
    d = np.random.rand(1, 2)
    e = np.full_like(d, 4)
    e1 = np.full((3, 2), 2)
    f = np.array([[[1,2],[3,4]],[[5,6],[7,8]]], dtype='int64')
    f1 = f[0, 0, 1]
    f2 = f[0, 0, :]
    # replace numbers
    f3 = np.array([[[1,2],[3,4]],[[5,6],[7,8]]], dtype='int64')
    f3[:,1,:] = [[9,9],[8,8]]
    print("f3", f3)
    # get info of array
    f4 = f.ndim
    f5 = f.shape
    f6 = f.dtype
    
    # example 1
    ex1 = np.ones((5, 5), dtype='int64')
    ex2 = np.zeros((3, 3), dtype='int64')
    ex2[1, 1] = 9
    ex1[1:4, 1:4] = ex2
    print("ex1", ex1)
    
    # signal
    # https://docs.scipy.org/doc/numpy/reference/routines.math.html
    h = np.random.rand(1, 10)
    h1 = np.cos(h)
    print("h1", h1)
    
    # linear algebra
    i1 = np.ones((2,2), dtype='int64')
    i2 = np.array([[1,2],[3,4]])
    i3 = i1 * i2
    i7 = np.dot(i1[0,:], i2[0,:])
    i8 = np.dot(i1[1,:], i2[1,:])
    print("i3", i3)
    i4 = np.ones((1, 3), dtype='int64')
    i5 = np.array([[1, 2], [3, 4], [5, 6]])
    i6 = np.matmul(i4, i5)
    print("i4", i4)
    print("i5", i5)
    print("i6", i6)
    i7 = np.zeros((1, 2))
    i8 = np.concatenate((i7, i5), axis=0)
    
    # min & max
    j1 = np.array([[1,2,3],[4,5,6]])
    j2 = np.min(j1[0,:])
    j3 = np.max(j1[0,:])
    j4 = np.max(j1)
    j5 = np.max(j1, axis=1)
    j6 = np.sum(j1[0,:])
    j7 = np.diff(j1, axis=0)
    j8 = np.sum(j1, axis=1)
    j9 = np.sum(j1)
    for j10, j11 in enumerate(j1):
        print("j10", j10)
        print("j11", j11)
        for j12, j13 in enumerate(j11):
            print("j12", j12)
            print("j13", j13)
    
    # reorganize
    k1 = np.array([[1,2,3,4],[5,6,7,8]])
    k2 = k1.reshape((4,2))
    print("k2", k2)
    # vertically stacking vectors
    k3 = np.array([1,2,3,4])
    k4 = np.array([5,6,7,8])
    k5 = np.vstack([k3, k3, k4, k4])
    print("k5", k5)
    # horizontal stack
    k6 = np.ones((2,4), dtype="int64")
    k7 = np.zeros((2,2), dtype="int64")
    k8 = np.hstack((k6, k7))
    print("k8", k8)
    
    # Miscellaneous
    filedata = np.genfromtxt('1.txt', delimiter='     ')
    print(filedata.astype('int64'))
    print(filedata>50)
    print(filedata[filedata>50])
    print(np.any(filedata > 380, axis=0))
    print(np.all(filedata > 380, axis=0))
    # all data more than 1 and less than 3
    print((filedata > 1) & (filedata < 3))
    # all data except more than 1 and less than 3
    print(~((filedata > 1) & (filedata < 3)))
    # index with a list in Numpy
    l = np.array([1,2,3,4,5,6,7,8,9])
    print("l[1,2,3]", l[[1,2,3]])
    
    # example 2
    m5 = np.arange(31)
    m = np.arange(1, 31)
    m1 = m.reshape(6, 5)
    print("m1", m1)
    # how to index
    m2 = m1[2:4, 0:2]
    m3 = m1[[0,1,2,3], [1,2,3,4]]
    m4 = m1[[0, 4, 5], 3:5]
    print("m2", m2)
    print("m3", m3)
    print("m4", m4)
    
    n = np.histogram([1,2,1], bins=[0,1,2,3])
    print("n", n)
    n1 = np.histogram(np.arange(4), bins=np.arange(5), density=True)
    print("n1", n1)
    n2 = np.arange(5)
    hist, bin_edges = np.histogram(n2, density=True)
    print("hist", hist)
    
    # funcs remain to be practiced: np.flatten, np.subtract
    