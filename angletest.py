#%%
import numpy as np
import matplotlib.pyplot as plt

#%%
positions = np.zeros(3, dtype=complex)
positions[0] -= 1
starty = np.random.uniform(-0.9,1)
positions[0] += 1j*starty
phi = np.tan(np.imag(positions[0])/np.real(positions[0]))
theta = np.pi/20

tphi = phi-theta
tphi *= -1
newphi = tphi +theta

fig, ax =plt.subplots()
mirror = np.linspace(-1,1, dtype=complex)
mirror += 1j*np.sin(theta)*mirror
ax.plot(np.array([-1,1]), np.zeros(2), color="black")
ax.plot(np.real(mirror), np.imag(mirror))
ax.set_ylim(-1,1)
positions[2] = 1*(np.cos(newphi) + 1j*np.sin(newphi))

ax.plot(np.real(positions), np.imag(positions))
ax.set_aspect(1)
print(np.arctan(np.imag(positions[2]))/np.real(positions[2]))
plt.show

# %%

# %%
