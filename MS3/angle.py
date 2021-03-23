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

x_resol = 20
y_resol = 20

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

data_ant0 = data['data'][0].reshape(10,Nr,Ns)
data_ant1 = data['data'][1].reshape(10,Nr,Ns)
data_ant2 = data['data'][2].reshape(10,Nr,Ns)
data_ant3 = data['data'][3].reshape(10,Nr,Ns)

data_bg0 = data['background'][0].reshape(10,Nr,Ns).mean(axis=0)
data_bg1 = data['background'][1].reshape(10,Nr,Ns).mean(axis=0)
data_bg2 = data['background'][2].reshape(10,Nr,Ns).mean(axis=0)
data_bg3 = data['background'][3].reshape(10,Nr,Ns).mean(axis=0)

data_ant0 -= data_bg0
data_ant1 -= data_bg1
data_ant2 -= data_bg2
data_ant3 -= data_bg3

# data_ant0 -= calib_ant0
# data_ant1 -= calib_ant1
# data_ant2 -= calib_ant2
# data_ant3 -= calib_ant3

# data_ant0 = np.fft.fftshift(data_ant0,axes=(2))
# data_ant1 = np.fft.fftshift(data_ant1,axes=(2))
# data_ant2 = np.fft.fftshift(data_ant2,axes=(2))
# data_ant3 = np.fft.fftshift(data_ant3,axes=(2))

maxes = np.empty((10,4),dtype='complex128')
for i in range(0,10):
    # C = data_ant0[i]
    # v = np.linspace(-1,1,C.shape[0])*c/(2*B)
    # d = np.linspace(0,1,C.shape[1])*( c*  np.pi* 1/Ts * 2 / ( 2 * 2*np.pi * f0 * Ns ) )
    # X, Y = np.meshgrid(d, v,)
    # print(X.shape,Y.shape)
    # fig, ax = plt.subplots()
    # ax.pcolormesh(X, Y, abs(C))
    # plt.savefig("def" + str(i) + ".png")
    # plt.close()

    # C = data_ant0[i]
    # v = np.linspace(-1,1,C.shape[0])*c/(2*B)
    # d = np.linspace(0,1,C.shape[1])*( c*  np.pi* 1/Ts * 2 / ( 2 * 2*np.pi * f0 * Ns ) )
    # X, Y = np.meshgrid(d, v,)
    # print(X.shape,Y.shape)
    # fig, ax = plt.subplots()
    # ax.pcolormesh(X, Y, abs(C))
    # plt.savefig("ghi" + str(i) + ".png")
    # plt.close()

    # C = data_ant0[i]
    # v = np.linspace(-1,1,C.shape[0])*c/(2*B)
    # d = np.linspace(0,1,C.shape[1])*( c*  np.pi* 1/Ts * 2 / ( 2 * 2*np.pi * f0 * Ns ) )
    # X, Y = np.meshgrid(d, v,)
    # print(X.shape,Y.shape)
    # fig, ax = plt.subplots()
    # ax.pcolormesh(X, Y, abs(C))
    # plt.savefig("jkl" + str(i) + ".png")
    # plt.close()

    # C = data_ant0[i]
    # v = np.linspace(-1,1,C.shape[0])*c/(2*B)
    # d = np.linspace(0,1,C.shape[1])*( c*  np.pi* 1/Ts * 2 / ( 2 * 2*np.pi * f0 * Ns ) )
    # X, Y = np.meshgrid(d, v,)
    # print(X.shape,Y.shape)
    # fig, ax = plt.subplots()
    # ax.pcolormesh(X, Y, abs(C))
    # plt.savefig("mno" + str(i) + ".png")
    # plt.close()


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

    max2 = 0
    argmax2 = (0,0)
    for j in range(x_resol):
        for k in range(y_resol):
            if(abs(data_ant2[i][j][k])>abs(max2)):
                argmax2 = (j,k)
                max2 = abs(data_ant2[i][j][k])

    max3 = 0
    argmax3 = (0,0)
    for j in range(x_resol):
        for k in range(y_resol):
            if(abs(data_ant3[i][j][k])>abs(max3)):
                argmax3 = (j,k)
                max3 = abs(data_ant3[i][j][k])

    tempmaxes = [max0,max1,max2,max3]
    args = [argmax0,argmax1,argmax2,argmax3]

    j = tempmaxes.index(max(tempmaxes))
    print("----")
    temp = np.array([data_ant0[i][args[j][0]][args[j][1]],data_ant1[i][args[j][0]][args[j][1]],data_ant2[i][args[j][0]][args[j][1]],data_ant3[i][args[j][0]][args[j][1]]],dtype='complex128')
    maxes[i,:] = temp

print(maxes)
Ux = np.linspace(-1,1,x_resol)
Uy = np.linspace(-1,1,y_resol)
for i in range(0,10):
    data_ant = data_ant0[i]
    X, Y = np.meshgrid(Ux, Uy,)
    print(X.shape,Y.shape)
    fig, ax = plt.subplots()
    ax.pcolormesh(X, Y, abs(frame))
    plt.savefig("abc" + str(i) + ".png")
    plt.close()
