import pandas as pd
import numpy as np
from scipy.integrate import odeint 
import os
import pickle
from pathlib import Path

"""Module define test data. Include sin, sign, real data from sensor and temperature data.
"""
# Pokud přidáte data, přidejte je i do listu data_all_names a data_all pod daty!!
script_dir = os.path.dirname(__file__)
data_folder = Path('test_data')
data_folder_path = script_dir /data_folder

#pokusna data 0 + date
rel_path0 = "daily-minimum-temperatures.csv"
abs_file_path0 = data_folder_path / rel_path0
data_ft_date = pd.read_csv(abs_file_path0, header=0, index_col=0)
data0t = data_ft_date.values
data0 = data0t.T[0]

#Pokusna data 1
dt = 0.3  #[sec]
t = np.arange(0,50,dt)
N = len(t)
data1 = np.sin(2 * np.pi / 10 * t)

#Pokusna data 2
dt=0.3  #[sec]
t = np.arange(0, 50, dt)
N = len(t)
data2a = np.sin(2 * np.pi / 10 * t)
data2 = np.sign(data2a)

# Pokusna data 3
def fdxdt(xx, t, u, Omega, eta, b0, b1):    # x=[x1 x2 ... xn] vektor hodnot n stavovych velicin
        dx1dt = -Omega0 ** 2 * xx[1]-b0 * u
        dx2dt = -2 * eta * Omega0 * xx[1] - b1 * u + xx[0]
        return(dx1dt,dx2dt)

dt = .1  #[sec]
t = np.arange(0,50,dt) ; N = len(t)  # delka dat
Npul = int(N/2)  # konverze na integer
u1 = np.sin(np.pi*t/4); u = np.sign(u1) # Vstup do systému
Omega0 = 10;  eta = .1;   b0 = Omega0**2;  b1 = 0
data3 = np.zeros(N)
x10 = 0
x20 = 0  # poc. podm
x0 = [x10, x20]


for i in range(0,N-1):
    tt = [t[i],t[i+1]]  # [t1 t2]
    xxx = odeint(fdxdt,x0,t,(u[i],Omega0,eta,b0,b1)) #returns x=[ [x1(t1) x2(t1)] [x1(t2) x2(t2)]]
    data3[i+1] = -xxx[1,1]
    x0 = xxx[1,:]

# Pokusná data 4
file_to_open = data_folder_path / "realna_data_klapky.txt"
data4 = np.loadtxt(file_to_open)
