import numpy as np
#Pokusna data 1
#dt = 0.01  #[sec]
#t = np.arange(0,100,dt)
#N = len(t)
#data1 = np.sin(2 * np.pi / 10 * t)
from numpy.random import randn

# generate some Gaussian values
data1 = np.random.randn(1000)*5 + 10

#Pokusna data 2
dt=0.1  #[sec]
t = np.arange(0, 1000, dt)
N = len(t)
data2a = np.sin(2 * np.pi / 10 * t)
data2 = np.sign(data2a)
