#%%
import numpy as np
import matplotlib.pyplot as plt
import spingen


class JoukowskiFlow(spingen.SpinGen):

    def __init__(self, gamma=0, nu=-0.22 +0.125j):
        super().__init__(gamma)
        self.nu = nu#-0.22 +0.125j
        print(f"np.abs(nu-1) {np.abs(self.nu - 1)}")
        self.alpha = 0
        self.a = np.abs(self.nu-1)

    def _particle_velocityfield(self, particles):

        return super()._particle_velocityfield(particles - self.nu)


    def coodmap(self, z):
        return (z) + 1/(z)

    def quickshow(self):
        fig, ax = plt.subplots(2,2)
        t = np.linspace(0,2*np.pi, 100, dtype=complex)
        dist = np.abs(self.nu - 1)
        t = self.nu + dist*(np.cos(np.real(t)) + 1j*np.sin(np.real(t)))
        #  t is the parametrized circle; so coodmap(t) should plot the aerofoil
        time = np.arange(0,4,self.dt)
        particles = np.empty((len(time),50),dtype=complex)
        particles[0] = -2.5 + 1j*np.linspace(-3,3,len(particles[0]),dtype=complex)
        
        for i in range(len(time)-1):
            particles[i+1] = particles[i] + self.dt*self._particle_velocityfield(particles[i])
        for axs in ax:
            for axss in axs:
                axss.plot(np.real(self.coodmap(t)), np.imag(self.coodmap(t)))
                axss.plot(np.real(t), np.imag(t), color="gray")
                axss.scatter(np.array([1]), np.zeros(1), color="black")
                axss.set_aspect(1)
        tograph = [0]*4
        for i in range(len(tograph)):
            tograph[i] = particles[int(i*len(particles)/len(tograph))]
            tograph[i] = self.coodmap(tograph[i])
        idl = int(len(particles)/4)
        ax[0,0].scatter(np.real(tograph[0]), np.imag(tograph[0]))
        ax[0,0].scatter(np.real(particles[0]), np.imag(particles[0]), color="gray")
        ax[0,1].scatter(np.real(tograph[1]), np.imag(tograph[1]))
        ax[0,1].scatter(np.real(particles[1*idl]), np.imag(particles[1*idl]), color="gray")
        ax[1,0].scatter(np.real(tograph[2]), np.imag(tograph[2]))
        ax[1,0].scatter(np.real(particles[2*idl]), np.imag(particles[2*idl]), color="gray")
        ax[1,1].scatter(np.real(tograph[3]), np.imag(tograph[3]))
        ax[1,1].scatter(np.real(particles[-1]), np.imag(particles[-1]), color="gray")
        
        plt.show()

    def sample_transforms(self):
        fig, axs = plt.subplots(1,3)
        nu = np.array([0.1+0.1j, 0.3 + 0.1j, self.nu])
        a = np.abs(1-nu)
        a[0] =0.7
        for i in range(3):
            t = np.linspace(0,2*np.pi, 1000)
            cylinder = (np.cos(t) +1j*np.sin(t))*a[i] + nu[i]
            axs[i].plot(np.real(self.coodmap(cylinder)), np.imag(self.coodmap(cylinder)), color="black")
            axs[i].plot(np.cos(t), np.sin(t), color="gray")
            axs[i].set_xlabel(f"$a={np.round(a[i], decimals=2)}$, $z_0={np.round(nu[i], decimals=2)}$")
        for ax in axs:
            ax.set_aspect(1)
        plt.show()
jkflow = JoukowskiFlow(0) # gamma = 0 for now
jkflow.sample_transforms()

#jkflow.quickshow()

""" def reverse_transform()
    # pseudocode for now
    trgarr = np.empty(len(particles), dtype=complex)
        a = 1
        b = -1*particles
        c = 1
        for i in range(len(particles)):
            zetas = np.roots([a,b[i],c])
            for root in zetas:
                if np.abs(root)<1:
                    trgarr[i] = root
        """


# %%
