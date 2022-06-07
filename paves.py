#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
fig, ax = plt.subplots(1)


DT = np.array([0.25, 0.1, 0.05, 0.025])
rk4 



t = np.linspace(0,2*np.pi,500)
ax.plot(np.sin(t),np.cos(t),color="black")
#rect = patches.Rectangle((-5, -5), 10, 10, linewidth=1, edgecolor="black", facecolor='none')

ax.set_xticks(np.linspace(-5,5,11))
ax.set_yticks(np.linspace(-5,5,11))
ax.plot(np.array([-5,5]), np.zeros(2)-5, color="black")
ax.plot(np.array([-5,5]), np.zeros(2)+5, color="black")
ax.plot(np.zeros(2)-5,np.array([-5,5]),  color="black")
ax.plot(np.zeros(2)+5,np.array([-5,5]),  color="black")
ax.set_aspect(1)
#ax.add_patch(rect)
ax.grid(1)

plt.show()
# %%
