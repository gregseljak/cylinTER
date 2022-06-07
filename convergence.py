#%%
import numpy as np
import matplotlib.pyplot as plt

# %%
gamma = 0
N = np.array([10,50,100,150,])
rsquared = np.array([0.11030722476519152,  0.08389182929233206, 0.08130878619205556, 0.077970028060041745, ])
fig, ax = plt.subplots(1)
fig.suptitle("Erreur RMS de la distribution des $S^{jk}_{t=2}$")
ax.set_xlabel("$ N$ (Pavés dans chaque axe de $H$)")
ax.set_ylabel("Erreur RMS")
ax.plot(N, rsquared,marker="x")
plt.show()
#%%
quit()
""""------------
N = 10
Calculated parameter: 14.296558529975501
Empirical poisson parameter: [14.47256177] : R^2 = 0.12648632480041194
Empirical poisson parameter: [14.47901913] : R^2 = 0.
Empirical poisson parameter: [14.3800394] : R^2 = 0.09168166110914705

N = 100
 Calculated parameter: 14.296558529975501
Empirical poisson parameter: [14.4219624] : R^2 = 0.08245440140686328
Empirical poisson parameter: [14.34924948] : R^2 = 
Empirical poisson parameter: [14.4170363] : R^2 = 0.08133447022885434

n = 150
Empirical poisson parameter: [14.42042768] : L^2 on analytic: = 0.08281528013435852
Empirical poisson parameter: [14.38354624] : L^2 on analytic: = 
Empirical poisson parameter: [14.412967] : L^2 on analytic: = 0.08069211418894479

N=50
Empirical poisson parameter: [14.40988485] : L^2 on analytic: = 0.08557321637371368
Empirical poisson parameter: [14.40214409] : L^2 on analytic: = 
Empirical poisson parameter: [14.45708474] : L^2 on analytic: = 0.08087000543938139
"""