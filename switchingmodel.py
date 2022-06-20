import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from switchingmodelfunc import arpoissontest

ts=arpoisson(alpha=-3, rho=0.1, sigma=0.4, n=100, p=0.6, q=0.8)
fig,ax=plt.subplots(1,3)
ax[0].plot(ts['etas'])
ax[1].plot(ts['zt'])
ax[2].plot(ts['y'])

print("hello world")