#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import poisson

x = np.arange(0,12)
base = poisson.pmf(x,4.5)
plt.plot(x, base)
plt.show()

# %%
def mypmf(x, mu):
    return poisson.pmf(x, mu)
retval = curve_fit(mypmf, x, base)
retval[0]
# %%
