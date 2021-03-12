import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import scipy.signal as scsa
import numpy as np
data = np.load('group1_marche_2.npz')
f0 = data[data.files[2]][0]
B = data[data.files[2]][1]
Ns = int(data[data.files[2]][2])
Nr = int(data[data.files[2]][3])
Ts = data[data.files[2]][4]
Tr = data[data.files[2]][5]
 
c = 3*10**8

bg = data[data.files[1]].reshape((Nr*Ns,1))
l = []

for i in range(0,data[data.files[0]].shape[0]):
    frame = (data[data.files[0]][i].reshape((Nr*Ns,1)) - bg).reshape(Nr,Ns)
    frame = frame - np.mean(frame)
    C = np.fft.fft2(frame,s = (200,200))
    C = np.fft.fftshift(C)
    print(C.shape)
    v = np.linspace(-1,1,C.shape[0])*c/(2*B)
    d = np.linspace(0,1,C.shape[1])*( c*  np.pi* 1/Ts * 2 / ( 2 * 2*np.pi * f0 * Ns ) )
    X, Y = np.meshgrid(d, v,)
    print(X.shape,Y.shape)
    fig, ax = plt.subplots()
    ax.pcolormesh(X, Y, abs(C))
    plt.savefig("abc" + str(i) + ".png")
    plt.close()