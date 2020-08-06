from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
import pandas as pd
import numpy as np

with open('D:\ceshidata.txt') as f:
    a = f.read().split('\n')
b = np.array([int(x) for x in a])
d = pd.DataFrame(data = b)
e = d.diff(1)
e.plot()
c = adfuller(b)
# print(c)
print(acorr_ljungbox(b,lags=2))
# import matplotlib.pyplot as plt
# plt.plot(b)
# plt.show()