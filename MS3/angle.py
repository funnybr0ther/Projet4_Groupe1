import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import scipy.signal as scsa
import numpy as np
data = np.load('angle.npz')
calib = np.load('calib0.npz')
f0 = data['chirp'][0]
B = data['chirp'][1]
Ns = int(data['chirp'][2])
Nr = int(data['chirp'][3])
Ts = data['chirp'][4]
Tr = data['chirp'][5]

x_resol = 200
y_resol = 200

c = 3*10**8
lamb = 2*np.pi * c/f0
d = 0.38*lamb

calib_ant0 = calib['data'][0].reshape(10,Nr,Ns)
calib_ant1 = calib['data'][1].reshape(10,Nr,Ns)
calib_ant2 = calib['data'][2].reshape(10,Nr,Ns)
calib_ant3 = calib['data'][3].reshape(10,Nr,Ns)

calib_bg0 = calib['background'][0].reshape(10,Nr,Ns).mean(axis=0)
calib_bg1 = calib['background'][1].reshape(10,Nr,Ns).mean(axis=0)
calib_bg2 = calib['background'][2].reshape(10,Nr,Ns).mean(axis=0)
calib_bg3 = calib['background'][3].reshape(10,Nr,Ns).mean(axis=0)

calib_ant0 = calib_ant0 - calib_bg0
calib_ant1 = calib_ant1 - calib_bg1
calib_ant2 = calib_ant2 - calib_bg2
calib_ant3 = calib_ant3 - calib_bg3

data_ant0 = data['data'][0].reshape(10,Nr,Ns) + 1j*data['data'][1].reshape(10,Nr,Ns)
data_ant1 = data['data'][2].reshape(10,Nr,Ns) + 1j*data['data'][3].reshape(10,Nr,Ns)

data_bg0 = data['background'][0].reshape(10,Nr,Ns).mean(axis=0) + 1j*data['background'][1].reshape(10,Nr,Ns).mean(axis=0)
data_bg1 = data['background'][2].reshape(10,Nr,Ns).mean(axis=0) + 1j*data['background'][3].reshape(10,Nr,Ns).mean(axis=0)

data_ant0 -= data_bg0
data_ant1 -= data_bg1

<<<<<<< HEAD
# data_ant0 -= calib_ant0
# data_ant1 -= calib_ant1
# data_ant2 -= calib_ant2
# data_ant3 -= calib_ant3

plt.plot(data_ant0[0].reshape(Ns*Nr,))
plt.plot(data_ant1[0].reshape(Ns*Nr,))
# plt.plot(data_ant2[0].reshape(Ns*Nr,))
# plt.plot(data_ant3[0].reshape(Ns*Nr,))

plt.xlim(0,20)
plt.show()
data_ant0 = np.fft.fft2(data_ant0,s = (x_resol,y_resol))
data_ant1 = np.fft.fft2(data_ant1,s = (x_resol,y_resol))
data_ant2 = np.fft.fft2(data_ant2,s = (x_resol,y_resol))
data_ant3 = np.fft.fft2(data_ant3,s = (x_resol,y_resol))


data_ant0 = np.fft.fftshift(data_ant0,axes=(2))
data_ant1 = np.fft.fftshift(data_ant1,axes=(2))
data_ant2 = np.fft.fftshift(data_ant2,axes=(2))
data_ant3 = np.fft.fftshift(data_ant3,axes=(2))

# data_ant0 = np.flip(data_ant0,axis=1)
# data_ant1 = np.flip(data_ant0,axis=1)
# data_ant2 = np.flip(data_ant0,axis=1)
# data_ant3 = np.flip(data_ant0,axis=1)


maxes = np.empty((10,4))
for i in range(0,10):
=======
data_ant0 = np.fft.fft2(data_ant0,s=(x_resol,y_resol))
data_ant1 = np.fft.fft2(data_ant1,s=(x_resol,y_resol))

maxes = np.empty((10,4),dtype='complex128')
for i in range(0,10):
    
>>>>>>> 1907ff21893a247c9f0e1ae98f4c38ec7dffb83f
    max0 = 0
    argmax0 = (0,0)
    for j in range(x_resol):
        for k in range(y_resol):
            if(abs(data_ant0[i][j][k])>abs(max0)):
                argmax0 = (j,k)
                max0 = abs(data_ant0[i][j][k])

    max1 = 0
    argmax1 = (0,0)
    for j in range(x_resol):
        for k in range(y_resol):
            if(abs(data_ant1[i][j][k])>abs(max1)):
                argmax1 = (j,k)
                max1 = abs(data_ant1[i][j][k])


<<<<<<< HEAD
    tempmaxes = [max0,max1,max2,max3]
    args = [argmax0,argmax1,argmax2,argmax3]
    print(args)
=======
    tempmaxes = [max0,max1]
    args = [argmax0,argmax1]

>>>>>>> 1907ff21893a247c9f0e1ae98f4c38ec7dffb83f
    j = tempmaxes.index(max(tempmaxes))
    print("----")
    temp = np.array([data_ant0[i][args[j][0]][args[j][1]],data_ant1[i][args[j][0]][args[j][1]]],dtype='complex128')
    maxes[i,:] = temp

print(maxes)
Ux = np.linspace(-1,1,x_resol)
Uy = np.linspace(-1,1,y_resol)
for i in range(0,10):
    data_ant = np.zeros((x_resol,y_resol))
    for ux in range(x_resol):
        for uy in range(y_resol):
            data_ant[ux][uy] = maxes[i][0] + maxes[i][1]* np.exp(-1j * d * Uy[uy])
            data_ant[ux][uy] += maxes[i][2] * np.exp(-1j * d * Ux[ux])
            data_ant[ux][uy] += maxes[i][3] * np.exp(-1j * d * Ux[ux] -1j * d * Uy[uy])
    X, Y = np.meshgrid(Ux, Uy,)
    print(X.shape,Y.shape)
    fig, ax = plt.subplots()
    ax.pcolormesh(X, Y, abs(data_ant))
    plt.savefig("abc" + str(i) + ".png")
    plt.close()
