#%%
import numpy as np
import matplotlib.pyplot as plt

allphi = np.linspace(-np.pi+0.01, np.pi, 5)
for bphi in allphi:
    positions = np.zeros(3, dtype=complex)
    #bphi = np.random.uniform(-np.pi,np.pi)
    positions[0] = np.cos(bphi) + 1j*np.sin(bphi)
    #positions[0] = np.random.uniform(-0.9,1) + 1j*np.random.uniform(-0.9,1)
    phi = np.arctan2(np.imag(positions[0]),np.real(positions[0]))
    theta = -np.pi/3
    tphi = phi - theta

    if phi > theta:
        tphi = phi - 2*theta
    else:
        tphi = phi - 2*theta

    print(f"tphi {tphi}")
    newphi = np.pi - tphi

    fig, ax =plt.subplots()
    t = np.linspace(-1,1, dtype=complex)
    mirror = np.array([-1*np.cos(theta),np.cos(theta)], dtype=complex)
    mirror +=  1j*np.array([-1*np.sin(theta), np.sin(theta)])
    ax.plot(np.array([-1,1]), np.zeros(2), color="black")
    ax.plot(np.real(mirror), np.imag(mirror), label='mirror')
    ax.set_ylim(-1.5,1.5)
    ax.set_xlim(-1.5, 1.5)
    ax.legend()
    positions[2] = 1*(np.cos(newphi) + 1j*np.sin(newphi))

    ax.plot(np.real(positions), np.imag(positions))
    ax.scatter(np.real(positions)[0], np.imag(positions)[0], color="green")
    ax.scatter(np.real(positions)[2], np.imag(positions)[2], color="red")

    ax.set_aspect(1)
    print(np.arctan(np.imag(positions[2]))/np.real(positions[2]))
    plt.show()


# %%

# %%
