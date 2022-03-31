#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

res = 10000
a = 1
frame = 2
ypx = np.linspace(-frame,frame,frame*res)
xpx = ypx.copy()
psi = np.linspace(-3,3,20)
radlen = int(res*np.sqrt(2))
streams = np.empty((len(psi), radlen), dtype=complex)
for i in range(len(psi)):
    radius = np.linspace(a+ 0.00000001, 10, radlen, dtype=np.double)
    for j in range(len(radius)):
        r = radius[j]
        y = psi[i]/(r**2-(a)**2)*r**2
        x = np.sqrt(r**2-y**2)
        streams[i,j] = x + 1j*y

def plot_stream(stream):
    x = np.append(-1*np.flip(np.real(stream)), np.real(stream))
    y = np.append(np.flip(np.imag(stream)), np.imag(stream))
    plt.plot(x, y, color="C0")
for stream in streams:
    plot_stream(stream)

t = np.linspace(0, 2*np.pi, 100)
plt.plot(a*np.cos(t), a*np.sin(t), linewidth=1, color="black")
plt.show()

# %%
