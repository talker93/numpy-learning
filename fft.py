#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 12:22:21 2021

@author: shanjiang
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftfreq, ifft

n = 1000
Lx = 100
omg = 2.0*np.pi/Lx

x = np.linspace(0, Lx, n)
y1 = 1.0*np.cos(5.0*omg*x)
y2 = 2.0*np.sin(10.0*omg*x)
y3 = 0.5*np.sin(20.0*omg*x)
print("x", x)

y = y1 + y2 + y3

freqs = fftfreq(n)

mask = freqs > 0

fft_vals = fft(y)

fft_theo = 2.0*np.abs(fft_vals/n)

plt.figure(1)
plt.title('Original Signal')
plt.plot(x, y, color='xkcd:salmon', label='original')
plt.legend()

plt.figure(2)
plt.plot(freqs, fft_vals, label="raw fft values")
plt.title("Raw FFT values - need more processing")
# plt.plot(freqs[mask], fft_theo[mask], label="true fft values")
# plt.title("True FFT values")
plt.show()