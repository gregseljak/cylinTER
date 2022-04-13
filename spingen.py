#%%
import numpy as np
import matplotlib.pyplot as plt
import flowgen

#%%
def binary_search(function, domain):
    ydomain = domain
    survey = function(ydomain)
    bigroots = np.where(survey[:-1]*survey[1:] <= 0)[0]
    if len(bigroots) == 0:
        print(" No roots seen in survey ")
        return np.array([])
    roots = np.empty(len(bigroots))
    for i in range(len(roots)):
        ymin = ydomain[bigroots[i]]
        ymax = ydomain[bigroots[i]+1]
        yguess = (ymin+ymax)/2
        
        cnt,threshold = 0, 1000
        while (abs(ymax-ymin) > 0.0001):
            yguess = (ymin+ymax)/2
            if function(yguess)*function(ymin) > 0:
                ymin = yguess
            if function(yguess)*function(ymax) > 0:
                ymax = yguess
            cnt += 1
            if cnt >= threshold:
                print(f"threshold {threshold} reached")
                break 
        roots[i] = yguess
    return roots

"""def func(x):
    return x**3-5*np.log(20*x)
domain = np.linspace(0.1,6,500)
roots = binary_search(func, domain)
plt.plot(domain, func(domain))
plt.scatter(roots, func(roots), color="tab:green")
print(f" roots: {roots} errs: {func(roots)}")"""
# %%

class SpinGen(flowgen.FlowGen):
    def __init__(self, gamma=10):
        self.v0 = 1
        self.xintval = np.array([-6,6])
        self.res = 1000
        self.a = 1
        self.radlen = int(self.res*np.sqrt(2))
        self.psi = self.psi_from_v(np.linspace(1.0001, 1.99, 20))
        self.phi = np.linspace(0,1,5)
        self.dt = 0.001
        self.streams = np.empty((len(self.psi), self.res), dtype=complex)
        self.equiphi = np.empty((len(self.psi), self.radlen), dtype=complex)
        self.gamma = gamma

    def equipsi(self,x,psi):

        def subordinate_equipsi(y):
            return psi - (self.v0*(y-self.a**2*y/(x**2+y**2) + self.gamma/(2*np.pi)*np.log(np.abs(x+1j*y))))   
        return subordinate_equipsi
    
    def vortex_equilines(self, ax, psi):
        
        xdomain = np.linspace(-2,2,100)
        ydomain = xdomain.copy()
        yvals = []
        for i in range(len(xdomain)):
            x = xdomain[i]
            onestream = self.equipsi(x,psi)
            roots = binary_search(onestream, ydomain)
            if len(roots) == 0:
                print(f" no roots found for psi={psi}")
                return np.zeros(len(xdomain))
            yvals.append(roots)
        maxlen = 0
        minlen = 100
        for rootset in yvals:
            if len(rootset) < minlen:
                minlen=len(rootset)
            if len(rootset) > maxlen:
                maxlen = len(rootset)
        #print(f"xdomain, minlen: {len(xdomain),minlen, maxlen}")
        finalyset = np.empty((len(xdomain), minlen))
        print(minlen)
        for i in range(len(xdomain)):
            for j in range(minlen):
                if i == 0:
                    finalyset[i,j] = (yvals[i])[j]
                else:
                    good_idx = np.argmin(np.abs(yvals[i]-finalyset[i-1,j]))
                    finalyset[i,j] = (yvals[i])[good_idx]
        self.equipsi = finalyset.transpose()



    
    def generate_spinstreams(self):
        fig, ax = plt.subplots(1)
        t = np.linspace(0,2*np.pi, 100)
        ax.plot(np.cos(t), np.sin(t), color="black")
        for psi in np.linspace(-1.9,1.9,20):
            self.vortex_equilines(ax, psi)
        #for psi in np.linspace(-1.1, -2, 4):
        #    self.vortex_streamlines(ax, psi)
        ax.legend()
        ax.set_aspect(1)
        plt.show()


flow = SpinGen(11)
flow.plot_spinstreams()
        
# %%
