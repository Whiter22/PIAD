import numpy as np
import scipy as sp
import pylab as py
from PIL import Image
import matplotlib.pyplot as plt


# def discrite(f, fs):
#     t = np.arange(0, 1, 1/fs)
#     s = np.sin(2*np.pi*f*t)
#     return t, s


# freq = 10 #Hz
# samples = [20, 21, 30, 45, 50, 100, 150, 200, 250, 1000]

# for sample in samples:
#     t, s = discrite(freq, sample)
#     plt.plot(t, s)
#     plt.show()

'''
4: twierdzenie Nyquista
5: aliasing
'''

img = np.array(Image.open('C:/Users/rubin/OneDrive/Pulpit/PIAD/lab4/img1.png'))
print(f"Shape:\nwidth: {img.shape[0]}\nheight: {img.shape[1]}")

values_per_px = len(img.getbands())
print(f"Pixels per pixel: {values_per_px}")